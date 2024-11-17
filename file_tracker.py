#!/usr/bin/python3
from bcc import BPF
import time
import os
import argparse
import threading
from datetime import datetime
import glob
from pathlib import Path
import pwd

# BPF program for current access tracking
bpf_text = """
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/mman.h>

struct data_t {
    u64 timestamp;
    u32 pid;
    u32 fd;
    s64 size;
    u64 offset;
    char comm[16];
    char syscall[8];
};

BPF_PERF_OUTPUT(events);

static __always_inline void set_syscall(struct data_t *data, const char *name) {
    #pragma unroll
    for (int i = 0; i < 8 && name[i]; i++) {
        data->syscall[i] = name[i];
    }
}

// Track open and openat for all processes
int syscall__open_enter(struct pt_regs *ctx, const char *filename, int flags) {
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.fd = 0;
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "open");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__openat_enter(struct pt_regs *ctx, int dirfd, const char *filename, int flags) {
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.fd = 0;
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "openat");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

// Track read for all processes
int syscall__read_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.fd = fd;
    data.size = count;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "read");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

// Track write for all processes
int syscall__write_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.fd = fd;
    data.size = count;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "write");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}
"""

class FileTracker:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.files_info = {}
        self.start_time = time.time()
        self.scan_checkpoint_files()

    def get_file_history(self, filepath):
        """Get file history from stat information"""
        try:
            stat = os.stat(filepath)
            username = pwd.getpwuid(stat.st_uid).pw_name

            history = {
                'path': filepath,
                'size': stat.st_size,
                'owner': username,
                'access_time': datetime.fromtimestamp(stat.st_atime),
                'modify_time': datetime.fromtimestamp(stat.st_mtime),
                'change_time': datetime.fromtimestamp(stat.st_ctime),
                'birth_time': datetime.fromtimestamp(stat.st_birthtime) if hasattr(stat, 'st_birthtime') else None,
                'accesses': []  # List to store live tracking data
            }
            return history
        except Exception as e:
            print(f"Error getting history for {filepath}: {e}")
            return None

    def scan_checkpoint_files(self):
        """Scan checkpoint directory for files and get their history"""
        try:
            for filepath in glob.glob(f"{self.checkpoint_dir}/*"):
                history = self.get_file_history(filepath)
                if history:
                    self.files_info[filepath] = history
                    self.print_file_history(filepath)
        except Exception as e:
            print(f"Error scanning checkpoint directory: {e}")

    def print_file_history(self, filepath):
        """Print history information for a file"""
        info = self.files_info.get(filepath)
        if not info:
            return

        filename = os.path.basename(filepath)
        print(f"\nFile: {filename}")
        print(f"Size: {info['size']} bytes")
        print(f"Owner: {info['owner']}")
        print(f"Timeline:")
        if info['birth_time']:
            print(f"  Created:  {info['birth_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Modified: {info['modify_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Changed:  {info['change_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Accessed: {info['access_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    def record_access(self, pid, syscall, filepath, size, timestamp):
        """Record a new access to a tracked file"""
        if filepath in self.files_info:
            access = {
                'timestamp': datetime.fromtimestamp(timestamp),
                'pid': pid,
                'syscall': syscall,
                'size': size
            }
            self.files_info[filepath]['accesses'].append(access)

            # Print access information immediately
            filename = os.path.basename(filepath)
            print(f"[LIVE] {datetime.now().strftime('%H:%M:%S')} - {filename}: {syscall} by PID {pid}, size: {size} bytes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Checkpoint directory to monitor")
    args = parser.parse_args()

    print(f"Analyzing checkpoint directory: {args.checkpoint_dir}")

    # Initialize file tracker
    tracker = FileTracker(args.checkpoint_dir)

    # Initialize BPF
    b = BPF(text=bpf_text)

    # Attach kprobes
    b.attach_kprobe(event="__x64_sys_read", fn_name="syscall__read_enter")
    b.attach_kprobe(event="__x64_sys_write", fn_name="syscall__write_enter")
    b.attach_kprobe(event="__x64_sys_open", fn_name="syscall__open_enter")
    b.attach_kprobe(event="__x64_sys_openat", fn_name="syscall__openat_enter")

    # Create closure to handle events
    def handle_event(cpu, data, size):
        event = b["events"].event(data)
        syscall = event.syscall.decode('utf-8', 'ignore').strip('\x00')

        try:
            proc_fd = f"/proc/{event.pid}/fd/{event.fd}"
            if os.path.exists(proc_fd):
                filepath = os.readlink(proc_fd)
                tracker.record_access(event.pid, syscall, filepath, event.size, event.timestamp)
        except (OSError, IOError):
            pass

    b["events"].open_perf_buffer(handle_event)

    print("\nMonitoring file access...")
    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nSummary of monitored files:")
        for filepath, info in tracker.files_info.items():
            filename = os.path.basename(filepath)
            print(f"\n{filename}:")
            print(f"Total accesses: {len(info['accesses'])}")
            if info['accesses']:
                print("Recent accesses:")
                for access in info['accesses'][-5:]:  # Show last 5 accesses
                    print(f"  {access['timestamp'].strftime('%H:%M:%S')} - {access['syscall']} "
                          f"(PID {access['pid']}, size: {access['size']} bytes)")

if __name__ == "__main__":
    main()
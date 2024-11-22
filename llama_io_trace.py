#!/usr/bin/python3
from bcc import BPF
import time
import os
import argparse
import threading
from datetime import datetime

# BPF program
bpf_text = """
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/fdtable.h>

// Data structure sent to user-space
struct data_t {
    u64 timestamp;
    u32 pid;
    u32 fd;
    s64 size;
    u64 offset;
    char fname[256];
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

static __always_inline void get_filename(const char *user_filename, char *kernel_buffer) {
    bpf_probe_read_user_str(kernel_buffer, 256, user_filename);
}

static __always_inline u64 get_file_offset(int fd) {
    struct task_struct *task;
    struct files_struct *files;
    struct fdtable *fdt;
    struct file **fdd;
    struct file *file;
    u64 pos = 0;

    task = (struct task_struct *)bpf_get_current_task();
    if (!task)
        return 0;

    files = task->files;
    if (!files)
        return 0;

    fdt = files->fdt;
    if (!fdt)
        return 0;

    if (fd >= fdt->max_fds)
        return 0;

    fdd = fdt->fd;
    if (!fdd)
        return 0;

    bpf_probe_read(&file, sizeof(file), &fdd[fd]);
    if (!file)
        return 0;

    bpf_probe_read(&pos, sizeof(pos), &file->f_pos);
    return pos;
}

int syscall__read_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = count;
    // The current offset before read
    data.offset = get_file_offset(fd);
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "read");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__write_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = count;
    // The current offset before write
    data.offset = get_file_offset(fd);
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "write");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__lseek_enter(struct pt_regs *ctx, int fd, off_t offset, int whence) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = 0;
    data.offset = offset;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "lseek");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__mmap_enter(struct pt_regs *ctx, void *addr, size_t length, int prot, int flags, int fd) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = length;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "mmap");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__fsync_enter(struct pt_regs *ctx, int fd) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
        
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "mmap");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}

int syscall__open_enter(struct pt_regs *ctx, const char *filename, int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = 0;     // Will be set in return probe
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "open");
    get_filename(filename, data.fname);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int syscall__openat_enter(struct pt_regs *ctx, int dirfd, const char *filename, int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = 0;     // Will be set in return probe
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "openat");
    get_filename(filename, data.fname);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// Track the returned file descriptor from open/openat
int syscall__open_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    int fd = PT_REGS_RC(ctx);
    if (fd >= 0) {
        struct data_t data = {};
        data.timestamp = bpf_ktime_get_ns();
        data.pid = pid;
        data.fd = fd;
        data.size = 0;
        
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        set_syscall(&data, "open_fd");
        events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}

int syscall__openat_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    int fd = PT_REGS_RC(ctx);
    if (fd >= 0) {
        struct data_t data = {};
        data.timestamp = bpf_ktime_get_ns();
        data.pid = pid;
        data.fd = fd;
        data.size = 0;
        
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        set_syscall(&data, "openat_fd");
        events.perf_submit(ctx, &data, sizeof(data));
    }
    return 0;
}

int syscall__pwrite64_enter(struct pt_regs *ctx, int fd, const void *buf, size_t count, loff_t pos) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = count;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "pwrite64");
    events.perf_submit(ctx, &data, sizeof(data));
    
    return 0;
}


"""

class IOTracker:
    def __init__(self):
        self.fd_to_name = {}
        self.fd_cache_time = {}
        self.start_time = None
        self.current_phase = "unknown"
        self.phase_stats = {}
        self.time_offset = time.time() - time.monotonic()

    # def format_size(self, size_bytes):
    #     """Format size in human readable format"""
    #     if size_bytes < 0:
    #         return f"error({size_bytes})"
    #     elif size_bytes < 1024:
    #         return f"{size_bytes}B"
    #     elif size_bytes < 1024 * 1024:
    #         return f"{size_bytes/1024:.2f}KB"
    #     else:
    #         return f"{size_bytes/1024/1024:.2f}MB"

    def get_wall_time(self, kernel_ns):
        """Convert kernel timestamp to wall clock time"""
        # Convert nanoseconds to seconds
        kernel_seconds = kernel_ns / 1e9

        # Calculate wall clock time
        wall_time = kernel_seconds + self.time_offset

        # Return formatted time string
        return datetime.fromtimestamp(wall_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def get_fd_path(self, pid, fd):
        """Get real file path from file descriptor"""
        if fd in self.fd_to_name:
            return self.fd_to_name[fd]

        try:
            fd_path = f"/proc/{pid}/fd/{fd}"
            if os.path.exists(fd_path):
                real_path = os.readlink(fd_path)  # Follow the symlink

                # Handle socket files
                if "socket:" in real_path:
                    inode = real_path.split("[")[1].split("]")[0]
                    return f"socket:[{inode}]"

                # Handle pipe files
                if "pipe:" in real_path:
                    inode = real_path.split("[")[1].split("]")[0]
                    return f"pipe:[{inode}]"

                # Store and return the real path
                self.fd_to_name[fd] = real_path
                self.fd_cache_time[fd] = time.time()
                return real_path

        except (OSError, IndexError) as e:
            return f"fd_{fd}"

    def refresh_fd_cache(self, pid):
        """Periodically refresh the FD cache"""
        current_time = time.time()

        # Clear old cache entries after 5 seconds
        old_fds = []
        for fd, cache_time in self.fd_cache_time.items():
            if current_time - cache_time > 5:
                old_fds.append(fd)
        for fd in old_fds:
            self.fd_to_name.pop(fd, None)  # Safe delete
            self.fd_cache_time.pop(fd, None)  # Safe delete

        # Rescan /proc/[pid]/fd directory
        try:
            fd_dir = f"/proc/{pid}/fd"
            if os.path.exists(fd_dir):
                for fd_name in os.listdir(fd_dir):
                    try:
                        fd = int(fd_name)
                        if fd not in self.fd_to_name:
                            path = self.get_fd_path(pid, fd)
                            if path:  # Only cache if we got a valid path
                                self.fd_to_name[fd] = path
                                self.fd_cache_time[fd] = current_time
                    except ValueError:
                        continue  # Skip non-integer fd names
        except Exception as e:
            pass

    def monitor_phase(self):
        """Monitor named pipe for phase updates"""
        pipe_path = "/tmp/llama_phase"

        while True:
            try:
                if not os.path.exists(pipe_path):
                    time.sleep(1)
                    continue

                with open(pipe_path, "r") as pipe:
                    while True:
                        phase = pipe.readline().strip()
                        if phase:
                            if phase != self.current_phase:
                                print(f"\n--- Switching to phase: {phase} ---")
                                self.current_phase = phase
                        else:
                            # If pipe is empty, sleep briefly to prevent busy waiting
                            time.sleep(0.1)
            except FileNotFoundError:
                self.current_phase = "unknown"  # Set default phase if pipe doesn't exist
                time.sleep(1)
            except Exception as e:
                print(f"Phase pipe error: {e}")
                self.current_phase = "error"    # Set error phase on exception
                time.sleep(1)

    def update_stats(self, phase, syscall, size, latency):
        """Update statistics for current phase"""
        if size < 0:  # Skip error returns
            return

        if phase not in self.phase_stats:
            self.phase_stats[phase] = {
                'read_count': 0,
                'write_count': 0,
                'read_bytes': 0,
                'write_bytes': 0
            }

        stats = self.phase_stats[phase]
        if syscall == 'read':
            stats['read_count'] += 1
            stats['read_bytes'] += size
        elif syscall == 'write':
            stats['write_count'] += 1
            stats['write_bytes'] += size

    def print_event(self, cpu, data, size):
        event = b["events"].event(data)
        syscall = event.syscall.decode('utf-8', 'ignore').strip('\x00')

        if event.size <= 0 and syscall not in ["lseek", "open", "openat", "open_fd", "openat_fd"]:
            return

        # Get real path (except for initial open calls which have the path in fname)
        if syscall in ["open", "openat"]:
            fname = event.fname.decode('utf-8', 'ignore').strip('\x00')
        else:
            # Refresh FD cache periodically
            self.refresh_fd_cache(event.pid)
            fname = self.get_fd_path(event.pid, event.fd)

        # Print formatted output based on syscall type
        timestamp = self.get_wall_time(event.timestamp)

        # Print formatted output with offset information
        if syscall == "lseek":
            print(f"[{self.current_phase:12}] "
                  f"[{timestamp}] "
                  f"{syscall:8} {fname:50} "
                  f"New Offset: {event.offset}")
        elif syscall in ["open", "openat"]:
            print(f"[{self.current_phase:12}] "
                  f"[{timestamp}] "
                  f"{syscall:8} {fname:50} "
                  f"Opening file")
        elif syscall in ["open_fd", "openat_fd"]:
            print(f"[{self.current_phase:12}] "
                  f"[{timestamp}] "
                  f"{syscall:8} {fname:50} "
                  f"Got FD: {event.fd}")
        else:
            print(f"[{self.current_phase:12}] "
                  f"[{timestamp}] "
                  f"{syscall:6} {fname:50} "
                  f"Offset: {event.offset} "
                  f"Size: {event.size} ")

    def print_summary(self):
        """Enhanced summary with access pattern statistics"""
        print("\nI/O Summary by Phase:")
        for phase, stats in self.phase_stats.items():
            print(f"\nPhase: {phase}")
            if stats['read_count'] > 0:
                print("Read operations:")
                print(f"  Count: {stats['read_count']}")
                print(f"  Total size: {self.format_size(stats['read_bytes'])}")

            if stats['write_count'] > 0:
                print("Write operations:")
                print(f"  Count: {stats['write_count']}")
                print(f"  Total size: {self.format_size(stats['write_bytes'])}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to trace")
    args = parser.parse_args()

    print("Starting I/O monitoring...")

    global b
    b = BPF(text=bpf_text.replace('LLAMA_PID', args.pid))

    # Attach kprobes
    b.attach_kprobe(event="__x64_sys_read", fn_name="syscall__read_enter")
    b.attach_kprobe(event="__x64_sys_write", fn_name="syscall__write_enter")
    b.attach_kprobe(event="__x64_sys_lseek", fn_name="syscall__lseek_enter")
    b.attach_kprobe(event="__x64_sys_mmap", fn_name="syscall__mmap_enter")
    b.attach_kprobe(event="__x64_sys_fsync", fn_name="syscall__fsync_enter")
    b.attach_kprobe(event="__x64_sys_open", fn_name="syscall__open_enter")
    b.attach_kretprobe(event="__x64_sys_open", fn_name="syscall__open_return")
    b.attach_kprobe(event="__x64_sys_openat", fn_name="syscall__openat_enter")
    b.attach_kretprobe(event="__x64_sys_openat", fn_name="syscall__openat_return")
    b.attach_kprobe(event="__x64_sys_pwrite64", fn_name="syscall__pwrite64_enter")

    # Create and setup tracker
    tracker = IOTracker()
    b["events"].open_perf_buffer(tracker.print_event)

    # Start phase monitoring thread
    phase_thread = threading.Thread(target=tracker.monitor_phase)
    phase_thread.daemon = True
    phase_thread.start()

    print(f"\nTracing I/O for PID {args.pid}... Ctrl+C to stop")
    print("PHASE        SYSCALL FILENAME                        SIZE                OFFSET             LATENCY")

    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nGenerating summary...")
        tracker.print_summary()

if __name__ == "__main__":
    main()
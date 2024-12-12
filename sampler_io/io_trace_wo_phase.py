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

int syscall__read_enter(struct pt_regs *ctx, struct kiocb *iocb, struct iov_iter *to) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != MONITOR_PID)
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
    if (pid != MONITOR_PID)
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
    if (pid != MONITOR_PID)
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

int syscall__openat_enter(struct pt_regs *ctx, int dirfd, const char *filename, int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != MONITOR_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = dirfd;     
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "openat");
    bpf_probe_read_user_str(data.fname, 256, filename);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int syscall__open_enter(struct pt_regs *ctx, const char *filename, int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != MONITOR_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = 0;
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "open");
    bpf_probe_read_user_str(data.fname, 256, filename);
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

int syscall__close_enter(struct pt_regs *ctx, int fd) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != MONITOR_PID)
        return 0;
    
    struct data_t data = {};
    data.timestamp = bpf_ktime_get_ns();
    data.pid = pid;
    data.fd = fd;
    data.size = 0;
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    set_syscall(&data, "close");
    
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

class IOTracker:
    def __init__(self):
        self.fd_to_name = {}
        self.fd_cache_time = {}
        self.start_time = None
        self.time_offset = time.time() - time.monotonic()
    def print_event(self, cpu, data, size):
        event = b["events"].event(data)
        syscall = event.syscall.decode('utf-8', 'ignore').strip('\x00')
        fname = event.fname.decode('utf-8', 'ignore').strip('\x00')

        if event.size <= 0 and syscall not in ["lseek", "open", "openat", "open_fd", "openat_fd"]:
            return

        # Print formatted output with offset information
        if syscall == "lseek":
            print(f"[{event.timestamp}] "
                  f"{syscall:8}"
                  f"New Offset: {event.offset}")
        elif syscall in ["openat", "open"]:
            print(f"[{event.timestamp}] "
                f"{syscall:8} {fname:50} ")
        elif syscall == "close":
            print(f"[{event.timestamp}] "
                  f"{syscall:6} /proc/{event.pid}/fd/{event.fd} "
            )
        else:
            print(f"[{event.timestamp}] "
                  f"{syscall:6} /proc/{event.pid}/fd/{event.fd} "
                  f"Offset: {event.offset} "
                  f"Size: {event.size} ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to trace")
    args = parser.parse_args()

    print("Starting I/O monitoring...")

    global b
    b = BPF(text=bpf_text.replace('MONITOR_PID', args.pid))

    # Attach kprobes
    b.attach_kprobe(event="ext4_file_read_iter", fn_name="syscall__read_enter")
    b.attach_kprobe(event="ext4_file_write_iter", fn_name="syscall__write_enter")
    b.attach_kprobe(event="__x64_sys_lseek", fn_name="syscall__lseek_enter")
    b.attach_kprobe(event="__x64_sys_open", fn_name="syscall__open_enter")
    b.attach_kprobe(event="__x64_sys_openat", fn_name="syscall__openat_enter")
    b.attach_kprobe(event="__x64_sys_close", fn_name="syscall__close_enter")

    # Create and setup tracker
    tracker = IOTracker()
    b["events"].open_perf_buffer(tracker.print_event)

    print(f"\nTracing I/O for PID {args.pid}... Ctrl+C to stop")
    print("SYSCALL        FILENAME                        SIZE                OFFSET             LATENCY")

    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nFinish.")

if __name__ == "__main__":
    main()
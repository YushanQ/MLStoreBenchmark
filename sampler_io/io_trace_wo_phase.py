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

BPF_PERF_OUTPUT(events);
#define DEFAULT_SUB_BUF_SIZE 255 
#define DEFAULT_SUB_BUF_LEN 16

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

BPF_PERCPU_ARRAY(data_map, struct data_t, 1);

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

static __always_inline void get_file_path(int fd, char *fname) {
    struct task_struct *task;
    struct files_struct *files;
    struct fdtable *fdt;
    struct file **fdd;
    struct file *file;
    
    task = (struct task_struct *)bpf_get_current_task();
    if (!task)
        return;

    files = task->files;
    if (!files)
        return;

    fdt = files->fdt;
    if (!fdt)
        return;

    if (fd >= fdt->max_fds)
        return;

    fdd = fdt->fd;
    if (!fdd)
        return;

    bpf_probe_read(&file, sizeof(file), &fdd[fd]);
    if (!file)
        return;
        
    struct dentry dtry;
    int nread = 0;
    int i = 0;
    bpf_probe_read(&dtry, sizeof(struct dentry), &file->f_path.dentry);
    bpf_probe_read_str(fname, DEFAULT_SUB_BUF_SIZE, dtry.d_name.name);
    
    nread++;
    for (i = 1; i < DEFAULT_SUB_BUF_LEN; i++) {
        if (dtry.d_parent != &dtry) {
            bpf_probe_read(&dtry, sizeof(struct dentry), dtry.d_parent);
            bpf_debug("read_enter: fpath=%d \\n", dtry.d_name.name); 
            // bpf_probe_read_str(path_buf->buffer[i], DEFAULT_SUB_BUF_SIZE, dtry.d_name.name);
            nread++;
        } else
            break;
    }
}


int syscall__read_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    int mark = 0;
    struct data_t *data = data_map.lookup(&mark);
    if (!data)
        return 0;
    
    data->timestamp = bpf_ktime_get_ns();
    data->pid = pid;
    data->fd = fd;
    data->size = count;
    data->offset = get_file_offset(fd);
    
    // generate filepath
    get_file_path(fd, data->fname);
    
    
    bpf_get_current_comm(&data->comm, sizeof(data->comm));
    set_syscall(data, "read");
    events.perf_submit(ctx, data, sizeof(*data));
    
    return 0;
}

int syscall__write_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    int mark = 0;
    struct data_t *data = data_map.lookup(&mark);
    if (!data)
        return 0;
    
    data->timestamp = bpf_ktime_get_ns();
    data->pid = pid;
    data->fd = fd;
    data->size = count;
    data->offset = get_file_offset(fd);
    
    // generate filepath
    
    
    bpf_get_current_comm(&data->comm, sizeof(data->comm));
    set_syscall(data, "read");
    events.perf_submit(ctx, data, sizeof(*data));
    
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

        # if event.size <= 0 and syscall not in ["lseek", "open", "openat", "open_fd", "openat_fd"]:
        #     return

        fname = event.fname.decode('utf-8', 'ignore').strip('\x00')

        # Print formatted output with offset information
        if syscall in ["read", "write"]:
            print(f"[{event.timestamp}] "
                  f"{syscall:6} {fname:50} "
                  f"Offset: {event.offset} "
                  f"Size: {event.size} ")


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

    # Create and setup tracker
    tracker = IOTracker()
    b["events"].open_perf_buffer(tracker.print_event)

    print(f"\nTracing I/O for PID {args.pid}... Ctrl+C to stop")

    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nFinish.")

if __name__ == "__main__":
    main()
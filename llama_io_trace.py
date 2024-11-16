#!/usr/bin/python3
from bcc import BPF
from time import sleep
import argparse
import os
import torch.profiler as profiler
import threading
import time
from pathlib import Path

# BPF program
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/fdtable.h>

// Structure to track file descriptors
struct fd_info_t {
    u32 pid;
    int fd;
    char name[256];
};

// Structure for detailed I/O data
struct io_data_t {
    u64 ts;
    u32 pid;
    u32 fd;
    u64 size;
    u64 offset;
    u64 flags;
    char comm[16];
};

// Maps to store data
BPF_HASH(start, u32, u64);
BPF_HASH(fd_info, u32, struct fd_info_t);
BPF_HASH(file_offsets, u32, u64);
BPF_PERF_OUTPUT(io_events);

// Helper: get filename for fd
static int get_filename(int fd, char *buf, size_t size) {
    struct task_struct *task;
    struct file *file;
    struct dentry *dentry;
    struct qstr d_name;
    
    task = (struct task_struct *)bpf_get_current_task();
    file = task->files->fdt->fd[fd];
    if (!file)
        return -1;
        
    dentry = file->f_path.dentry;
    if (!dentry)
        return -1;
        
    d_name = dentry->d_name;
    if (!d_name.name)
        return -1;
        
    bpf_probe_read_kernel_str(buf, size, d_name.name);
    return 0;
}

// Track file opens
int trace_open_enter(struct pt_regs *ctx, const char __user *filename, int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
        
    struct fd_info_t fd_data = {};
    fd_data.pid = pid;
    bpf_probe_read_user_str(fd_data.name, sizeof(fd_data.name), filename);
    
    u64 id = bpf_get_current_pid_tgid();
    fd_info.update(&id, &fd_data);
    return 0;
}

int trace_open_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
        
    int fd = PT_REGS_RC(ctx);
    if (fd >= 0) {
        u64 id = bpf_get_current_pid_tgid();
        struct fd_info_t *fd_data = fd_info.lookup(&id);
        if (fd_data) {
            fd_data->fd = fd;
            // Keep the entry updated
            fd_info.update(&id, fd_data);
        }
    }
    return 0;
}

// Track read operations
int trace_read_enter(struct pt_regs *ctx, int fd) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 ts = bpf_ktime_get_ns();
    u64 id = bpf_get_current_pid_tgid();
    start.update(&id, &ts);
    return 0;
}

int trace_read_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 id = bpf_get_current_pid_tgid();
    u64 *tsp = start.lookup(&id);
    if (!tsp)
        return 0;
    
    struct io_data_t data = {};
    data.ts = bpf_ktime_get_ns() - *tsp;
    data.pid = pid;
    data.size = PT_REGS_RC(ctx);
    
    // Get current offset
    struct fd_info_t *fd_data = fd_info.lookup(&id);
    if (fd_data) {
        data.fd = fd_data->fd;
        u64 *offset = file_offsets.lookup(&id);
        if (offset)
            data.offset = *offset;
    }
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    io_events.perf_submit(ctx, &data, sizeof(data));
    
    // Update offset
    if (data.size > 0) {
        u64 new_offset = data.offset + data.size;
        file_offsets.update(&id, &new_offset);
    }
    
    start.delete(&id);
    return 0;
}

// Track write operations
int trace_write_enter(struct pt_regs *ctx, int fd) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 ts = bpf_ktime_get_ns();
    u64 id = bpf_get_current_pid_tgid();
    start.update(&id, &ts);
    return 0;
}

int trace_write_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 id = bpf_get_current_pid_tgid();
    u64 *tsp = start.lookup(&id);
    if (!tsp)
        return 0;
    
    struct io_data_t data = {};
    data.ts = bpf_ktime_get_ns() - *tsp;
    data.pid = pid;
    data.size = PT_REGS_RC(ctx);
    
    struct fd_info_t *fd_data = fd_info.lookup(&id);
    if (fd_data) {
        data.fd = fd_data->fd;
        u64 *offset = file_offsets.lookup(&id);
        if (offset)
            data.offset = *offset;
    }
    
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    io_events.perf_submit(ctx, &data, sizeof(data));
    
    if (data.size > 0) {
        u64 new_offset = data.offset + data.size;
        file_offsets.update(&id, &new_offset);
    }
    
    start.delete(&id);
    return 0;
}

// Track lseek
int trace_lseek_enter(struct pt_regs *ctx, int fd, off_t offset, int whence) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 id = bpf_get_current_pid_tgid();
    file_offsets.update(&id, &offset);
    return 0;
}
"""

class IOStats:
    def __init__(self):
        self.io_events = []
        self.phase = "unknown"
        self.phase_start = time.time()
        # Map to store file descriptors to names
        self.fd_to_name = {}

    def add_io(self, event, filename):
        self.io_events.append({
            'phase': self.phase,
            'timestamp': time.time(),
            'filename': filename,
            'size': event.size,
            'offset': event.offset,
            'latency': event.ts,
            'operation': 'read' if event.comm.startswith('read') else 'write'
        })

    def set_phase(self, phase):
        if phase != self.phase:
            print(f"\nPhase change: {phase} (previous: {self.phase})")
            self.phase = phase
            self.phase_start = time.time()

    def monitor_phase_pipe(self):
        """Monitor named pipe for phase updates from PyTorch"""
        pipe_path = "/tmp/llama_phase"

        while True:
            if not os.path.exists(pipe_path):
                print(f"Waiting for phase pipe at {pipe_path}...")
                time.sleep(1)
                continue

            try:
                with open(pipe_path, "r") as pipe:
                    while True:
                        phase = pipe.readline().strip()
                        if phase:
                            self.set_phase(phase)
            except Exception as e:
                print(f"Phase pipe error: {e}")
                time.sleep(1)

    def print_summary(self):
        print("\nI/O Summary by Phase:")
        phases = set(e['phase'] for e in self.io_events)

        for phase in phases:
            phase_events = [e for e in self.io_events if e['phase'] == phase]
            print(f"\nPhase: {phase}")
            print(f"Total I/O operations: {len(phase_events)}")

            reads = [e for e in phase_events if e['operation'] == 'read']
            writes = [e for e in phase_events if e['operation'] == 'write']

            if reads:
                print("\nRead operations:")
                print(f"  Count: {len(reads)}")
                print(f"  Total size: {sum(e['size'] for e in reads)/1024/1024:.2f}MB")
                print(f"  Avg latency: {sum(e['latency'] for e in reads)/len(reads)/1000000:.2f}ms")
                print("\nMost accessed files (reads):")
                files = {}
                for e in reads:
                    files[e['filename']] = files.get(e['filename'], 0) + e['size']
                for f, size in sorted(files.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {f}: {size/1024/1024:.2f}MB")

            if writes:
                print("\nWrite operations:")
                print(f"  Count: {len(writes)}")
                print(f"  Total size: {sum(e['size'] for e in writes)/1024/1024:.2f}MB")
                print(f"  Avg latency: {sum(e['latency'] for e in writes)/len(writes)/1000000:.2f}ms")
                print("\nMost accessed files (writes):")
                files = {}
                for e in writes:
                    files[e['filename']] = files.get(e['filename'], 0) + e['size']
                for f, size in sorted(files.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {f}: {size/1024/1024:.2f}MB")

def print_event(cpu, data, size):
    event = b["io_events"].event(data)
    filename = stats.fd_to_name.get(event.fd, f"fd_{event.fd}")

    if event.size > 0:
        print(f"[{stats.phase}] {event.comm} {filename} "
              f"Size: {event.size/1024/1024:.2f}MB "
              f"Offset: {event.offset} "
              f"Latency: {event.ts/1000000:.2f}ms")
        stats.add_io(event, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to trace")
    args = parser.parse_args()

    # Initialize monitoring
    stats = IOStats()

    # Start phase monitoring thread
    phase_thread = threading.Thread(target=stats.monitor_phase_pipe)
    phase_thread.daemon = True
    phase_thread.start()

    # Initialize BPF
    bpf_text = bpf_text.replace('LLAMA_PID', args.pid)
    b = BPF(text=bpf_text)

    # Attach probes
    b.attach_kprobe(event="__x64_sys_read", fn_name="trace_read_enter")
    b.attach_kretprobe(event="__x64_sys_read", fn_name="trace_read_return")
    b.attach_kprobe(event="__x64_sys_write", fn_name="trace_write_enter")
    b.attach_kretprobe(event="__x64_sys_write", fn_name="trace_write_return")
    b.attach_kprobe(event="__x64_sys_open", fn_name="trace_open_enter")
    b.attach_kretprobe(event="__x64_sys_open", fn_name="trace_open_return")
    b.attach_kprobe(event="__x64_sys_lseek", fn_name="trace_lseek_enter")

    # Set up event monitoring
    b["io_events"].open_perf_buffer(print_event)

    print(f"\nTracing LLaMa I/O (PID: {args.pid})... Ctrl+C to end.")
    print("Waiting for phase updates from PyTorch training...")

    try:
        while True:
            b.perf_buffer_poll()
    except KeyboardInterrupt:
        stats.print_summary()
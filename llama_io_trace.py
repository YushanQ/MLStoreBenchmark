#!/usr/bin/python3
from bcc import BPF
import time
import os
import argparse
import threading

# BPF program
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/sched.h>

struct data_t {
    u64 ts;
    u32 pid;
    u32 fd;
    u64 size;
    u64 offset;
    char comm[16];
    char syscall[8];
};

// Track offset for each file descriptor
struct fd_info {
    u64 offset;
    u64 size;
};

BPF_HASH(start, u32, u64);
BPF_HASH(fds, u32, u32);
BPF_HASH(fd_offsets, u32, struct fd_info);
BPF_PERF_OUTPUT(events);

static __always_inline void set_syscall(struct data_t *data, const char *name) {
    #pragma unroll
    for (int i = 0; i < 8 && name[i]; i++) {
        data->syscall[i] = name[i];
    }
}

int syscall__read_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 ts = bpf_ktime_get_ns();
    start.update(&pid, &ts);
    fds.update(&pid, &fd);
    
    // Get existing fd_info or initialize new one
    struct fd_info info = {};
    struct fd_info *existing_info = fd_offsets.lookup(&fd);
    if (existing_info) {
        info = *existing_info;
    }
    info.size = count;  // Store requested size
    fd_offsets.update(&fd, &info);
    
    return 0;
}

int syscall__read_return(struct pt_regs *ctx) {
    struct data_t data = {};
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    if (pid != LLAMA_PID)
        return 0;
    
    u64 *tsp = start.lookup(&pid);
    u32 *fdp = fds.lookup(&pid);
    if (!tsp || !fdp)
        return 0;
        
    int fd = *fdp;
    struct fd_info *info = fd_offsets.lookup(&fd);
    if (!info)
        return 0;
    
    data.ts = bpf_ktime_get_ns() - *tsp;
    data.pid = pid;
    data.fd = fd;
    data.offset = info->offset;
    data.size = PT_REGS_RC(ctx);  // Actual bytes read
    
    // Update offset for next operation
    if (data.size > 0) {
        info->offset += data.size;
        fd_offsets.update(&fd, info);
    }
    
    set_syscall(&data, "read");
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events.perf_submit(ctx, &data, sizeof(data));
    
    start.delete(&pid);
    fds.delete(&pid);
    return 0;
}

int syscall__write_enter(struct pt_regs *ctx, int fd, void *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 ts = bpf_ktime_get_ns();
    start.update(&pid, &ts);
    fds.update(&pid, &fd);
    
    struct fd_info info = {};
    struct fd_info *existing_info = fd_offsets.lookup(&fd);
    if (existing_info) {
        info = *existing_info;
    }
    info.size = count;
    fd_offsets.update(&fd, &info);
    
    return 0;
}

int syscall__write_return(struct pt_regs *ctx) {
    struct data_t data = {};
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    if (pid != LLAMA_PID)
        return 0;
    
    u64 *tsp = start.lookup(&pid);
    u32 *fdp = fds.lookup(&pid);
    if (!tsp || !fdp)
        return 0;
        
    int fd = *fdp;
    struct fd_info *info = fd_offsets.lookup(&fd);
    if (!info)
        return 0;
    
    data.ts = bpf_ktime_get_ns() - *tsp;
    data.pid = pid;
    data.fd = fd;
    data.offset = info->offset;
    data.size = PT_REGS_RC(ctx);
    
    if (data.size > 0) {
        info->offset += data.size;
        fd_offsets.update(&fd, info);
    }
    
    set_syscall(&data, "write");
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events.perf_submit(ctx, &data, sizeof(data));
    
    start.delete(&pid);
    fds.delete(&pid);
    return 0;
}

int syscall__lseek_enter(struct pt_regs *ctx, int fd, long offset, unsigned int whence) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (pid != LLAMA_PID)
        return 0;
    
    u64 ts = bpf_ktime_get_ns();
    start.update(&pid, &ts);
    fds.update(&pid, &fd);
    return 0;
}

int syscall__lseek_return(struct pt_regs *ctx) {
    struct data_t data = {};
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    if (pid != LLAMA_PID)
        return 0;
    
    u64 *tsp = start.lookup(&pid);
    u32 *fdp = fds.lookup(&pid);
    if (!tsp || !fdp)
        return 0;
        
    int fd = *fdp;
    struct fd_info *info = fd_offsets.lookup(&fd);
    if (!info) {
        info = &(struct fd_info){};
    }
    
    data.ts = bpf_ktime_get_ns() - *tsp;
    data.pid = pid;
    data.fd = fd;
    
    // Update the offset based on lseek return value
    long new_offset = PT_REGS_RC(ctx);
    if (new_offset >= 0) {
        info->offset = new_offset;
        fd_offsets.update(&fd, info);
        data.offset = new_offset;
    } else {
        data.offset = info->offset;
    }
    
    data.size = 0;  // No data transferred in seek
    set_syscall(&data, "lseek");
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events.perf_submit(ctx, &data, sizeof(data));
    
    start.delete(&pid);
    fds.delete(&pid);
    return 0;
}
"""

class IOTracker:
    def __init__(self):
        self.fd_to_name = {}
        self.fd_cache_time = {}
        self.start_time = time.time()
        self.current_phase = "unknown"
        self.phase_stats = {}
        self.sequential_count = 0
        self.random_count = 0
        self.last_offset = {}  # Track last offset per FD

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
            if not os.path.exists(pipe_path):
                print(f"Waiting for phase pipe at {pipe_path}...")
                time.sleep(1)
                continue

            try:
                with open(pipe_path, "r") as pipe:
                    while True:
                        phase = pipe.readline().strip()
                        if phase:
                            if phase != self.current_phase:
                                print(f"\n--- Switching to phase: {phase} ---")
                                self.current_phase = phase
            except Exception as e:
                print(f"Phase pipe error: {e}")
                time.sleep(1)

    def is_sequential_access(self, fd, offset, size):
        """Determine if access is sequential based on last offset"""
        if fd not in self.last_offset:
            self.last_offset[fd] = offset
            return True

        expected_offset = self.last_offset[fd]
        self.last_offset[fd] = offset + size

        # Allow small gaps (e.g., alignment padding)
        return abs(offset - expected_offset) <= 4096

    def update_access_pattern(self, fd, offset, size):
        """Update sequential vs random access statistics"""
        if self.is_sequential_access(fd, offset, size):
            self.sequential_count += 1
        else:
            self.random_count += 1

    def update_stats(self, phase, syscall, size, latency):
        """Update statistics for current phase"""
        if phase not in self.phase_stats:
            self.phase_stats[phase] = {
                'read_count': 0,
                'write_count': 0,
                'read_bytes': 0,
                'write_bytes': 0,
                'read_latency': 0,
                'write_latency': 0
            }

        stats = self.phase_stats[phase]
        if syscall == 'read':
            stats['read_count'] += 1
            stats['read_bytes'] += size
            stats['read_latency'] += latency
        elif syscall == 'write':
            stats['write_count'] += 1
            stats['write_bytes'] += size
            stats['write_latency'] += latency

    def print_event(self, cpu, data, size):
        event = b["events"].event(data)

        if event.size <= 0 and event.syscall.decode('utf-8', 'ignore') != "lseek":
            return

        # Refresh FD cache periodically
        self.refresh_fd_cache(event.pid)

        # Get real path
        fname = self.get_fd_path(event.pid, event.fd)
        syscall = event.syscall.decode('utf-8', 'ignore')

        # Update access pattern statistics
        if syscall in ['read', 'write']:
            self.update_access_pattern(event.fd, event.offset, event.size)

        # Update phase statistics
        if syscall in ['read', 'write']:
            self.update_stats(self.current_phase, syscall, event.size, event.ts)

        # Print formatted output with offset information
        if syscall == "lseek":
            print(f"[{self.current_phase:12}] "
                  f"{syscall:6} {fname:50} "
                  f"New Offset: {event.offset} "
                  f"Latency: {event.ts/1000000:.2f}ms")
        else:
            print(f"[{self.current_phase:12}] "
                  f"{syscall:6} {fname:50} "
                  f"Size: {event.size/1024/1024:.2f}MB "
                  f"Offset: {event.offset} "
                  f"Latency: {event.ts/1000000:.2f}ms")

    def print_summary(self):
        """Enhanced summary with access pattern statistics"""
        print("\nI/O Summary by Phase:")
        for phase, stats in self.phase_stats.items():
            print(f"\nPhase: {phase}")
            if stats['read_count'] > 0:
                print("Read operations:")
                print(f"  Count: {stats['read_count']}")
                print(f"  Total size: {stats['read_bytes']/1024/1024:.2f}MB")
                print(f"  Avg latency: {stats['read_latency']/stats['read_count']/1000000:.2f}ms")

            if stats['write_count'] > 0:
                print("Write operations:")
                print(f"  Count: {stats['write_count']}")
                print(f"  Total size: {stats['write_bytes']/1024/1024:.2f}MB")
                print(f"  Avg latency: {stats['write_latency']/stats['write_count']/1000000:.2f}ms")

        total_ops = self.sequential_count + self.random_count
        if total_ops > 0:
            seq_percent = (self.sequential_count / total_ops) * 100
            rand_percent = (self.random_count / total_ops) * 100
            print("\nAccess Pattern Analysis:")
            print(f"Sequential accesses: {self.sequential_count} ({seq_percent:.1f}%)")
            print(f"Random accesses: {self.random_count} ({rand_percent:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", help="Process ID to trace")
    args = parser.parse_args()

    print("Starting I/O monitoring...")

    global b
    b = BPF(text=bpf_text.replace('LLAMA_PID', args.pid))

    # Attach kprobes
    b.attach_kprobe(event="__x64_sys_read", fn_name="syscall__read_enter")
    b.attach_kretprobe(event="__x64_sys_read", fn_name="syscall__read_return")
    b.attach_kprobe(event="__x64_sys_write", fn_name="syscall__write_enter")
    b.attach_kretprobe(event="__x64_sys_write", fn_name="syscall__write_return")
    b.attach_kprobe(event="__x64_sys_lseek", fn_name="syscall__lseek_enter")
    b.attach_kretprobe(event="__x64_sys_lseek", fn_name="syscall__lseek_return")

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
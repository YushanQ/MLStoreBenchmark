#!/usr/bin/env python3
from bcc import BPF, USDT
import os
import sys
import time
import subprocess
import signal
import atexit

# eBPF 程序代码
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

struct syscall_event_t {
    u32 pid;
    u64 timestamp;
    char comm[TASK_COMM_LEN];
    int syscall_id;
    char syscall_name[32];
    u64 arg0;
    u64 arg1;
    u64 arg2;
};

BPF_PERF_OUTPUT(events);
BPF_HASH(target_pids, u32, u32);

int trace_python_function_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u32 value = 1;
    target_pids.update(&pid, &value);
    return 0;
}

int trace_python_function_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    target_pids.delete(&pid);
    return 0;
}

static inline void trace_syscall(struct pt_regs *ctx, int syscall_id, const char *name) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    u32 *exists = target_pids.lookup(&pid);
    if (!exists)
        return;

    struct syscall_event_t event = {};
    event.pid = pid;
    event.timestamp = bpf_ktime_get_ns();
    event.syscall_id = syscall_id;
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    bpf_probe_read_str(&event.syscall_name, sizeof(event.syscall_name), name);
    
    event.arg0 = PT_REGS_PARM1(ctx);
    event.arg1 = PT_REGS_PARM2(ctx);
    event.arg2 = PT_REGS_PARM3(ctx);
    
    events.perf_submit(ctx, &event, sizeof(event));
}

int trace_open(struct pt_regs *ctx) {
    trace_syscall(ctx, 2, "open");
    return 0;
}

int trace_read(struct pt_regs *ctx) {
    trace_syscall(ctx, 0, "read");
    return 0;
}

int trace_close(struct pt_regs *ctx) {
    trace_syscall(ctx, 3, "close");
    return 0;
}

int trace_mmap(struct pt_regs *ctx) {
    trace_syscall(ctx, 9, "mmap");
    return 0;
}

int trace_fsync(struct pt_regs *ctx) {
    trace_syscall(ctx, 74, "fsync");
    return 0;
}
"""

class TracerManager:
    def __init__(self, script_path, function_name):
        self.script_path = script_path
        self.function_name = function_name
        self.target_process = None
        self.bpf = None

        # 注册退出处理
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start_target_process(self):
        """启动目标Python脚本"""
        self.target_process = subprocess.Popen(['python3', '-X', 'utf8', self.script_path],
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
        print(f"目标进程已启动，PID: {self.target_process.pid}")
        return self.target_process.pid

    def print_event(self, cpu, data, size):
        """处理eBPF事件"""
        event = self.bpf["events"].event(data)
        timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp / 1000000000))
        syscall_name = event.syscall_name.decode('utf-8', 'replace')

        print(f"[{timestamp}] PID: {event.pid} "
              f"SYSCALL: {syscall_name} ({event.syscall_id}) "
              f"ARGS: [0x{event.arg0:x}, 0x{event.arg1:x}, 0x{event.arg2:x}] "
              f"COMM: {event.comm.decode('utf-8', 'replace')}")

    def start_tracing(self):
        """启动跟踪"""
        try:
            # 启动目标进程
            target_pid = self.start_target_process()

            # 加载eBPF程序
            self.bpf = BPF(text=bpf_text)

            # 附加USDT探针
            u = USDT(pid=target_pid)
            u.enable_probe("function__entry", "trace_python_function_entry")
            u.enable_probe("function__return", "trace_python_function_return")

            # 附加系统调用跟踪器
            self.bpf.attach_kprobe(event="__x64_sys_open", fn_name="trace_open")
            self.bpf.attach_kprobe(event="__x64_sys_read", fn_name="trace_read")
            self.bpf.attach_kprobe(event="__x64_sys_close", fn_name="trace_close")
            self.bpf.attach_kprobe(event="__x64_sys_mmap", fn_name="trace_mmap")
            self.bpf.attach_kprobe(event="__x64_sys_fsync", fn_name="trace_fsync")

            # 设置事件回调
            self.bpf["events"].open_perf_buffer(self.print_event)

            print(f"开始跟踪函数 {self.function_name} 的系统调用...")
            print("监控的系统调用: open, read, close, mmap, fsync")
            print("按 Ctrl+C 停止跟踪")

            # 主循环
            while True:
                self.bpf.perf_buffer_poll()

                # 检查目标进程是否还在运行
                if self.target_process.poll() is not None:
                    stdout, stderr = self.target_process.communicate()
                    print("\n目标进程已结束")
                    if stdout:
                        print("进程输出:", stdout.decode())
                    if stderr:
                        print("进程错误:", stderr.decode())
                    break

        except Exception as e:
            print(f"发生错误: {e}")
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.target_process and self.target_process.poll() is None:
            self.target_process.terminate()
            self.target_process.wait()
        if self.bpf:
            self.bpf.cleanup()

    def signal_handler(self, sig, frame):
        """处理信号"""
        print("\n接收到终止信号，正在清理...")
        self.cleanup()
        sys.exit(0)

def main():
    if len(sys.argv) != 3:
        print("Usage: %s <target_script.py> <function_name>" % sys.argv[0])
        sys.exit(1)

    script_path = sys.argv[1]
    function_name = sys.argv[2]

    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本文件 {script_path}")
        sys.exit(1)

    tracer = TracerManager(script_path, function_name)
    tracer.start_tracing()

if __name__ == "__main__":
    main()
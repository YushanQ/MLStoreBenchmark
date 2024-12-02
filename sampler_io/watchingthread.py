import threading
import os
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Optional

class FileDescriptorMonitor(threading.Thread):
    def __init__(self, pid: Optional[int] = None, interval: float = 1.0):
        """
        初始化文件描述符监控线程

        Args:
            pid: 要监控的进程ID，默认为当前进程
            interval: 监控间隔时间（秒）
        """
        super().__init__(daemon=True)
        self.pid = pid or os.getpid()
        self.interval = interval
        self.fd_map: Dict[int, str] = {}
        self.running = True
        self.lock = threading.Lock()
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        self.logger = logging.getLogger('FDMonitor')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_fd_path(self, fd: int) -> Optional[str]:
        """
        获取文件描述符对应的文件路径

        Args:
            fd: 文件描述符

        Returns:
            文件路径或None（如果无法获取）
        """
        try:
            proc = psutil.Process(self.pid)
            for open_file in proc.open_files():
                if open_file.fd == fd:
                    return open_file.path
            return None
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    def get_fd_map(self) -> Dict[int, str]:
        """
        获取当前文件描述符映射的副本

        Returns:
            文件描述符到文件路径的映射字典
        """
        with self.lock:
            return self.fd_map.copy()

    def run(self):
        """监控线程的主循环"""
        self.logger.info(f"开始监控进程 {self.pid} 的文件描述符")

        prev_fds = set()
        while self.running:
            try:
                # 获取当前所有的文件描述符
                proc = psutil.Process(self.pid)
                current_fds = {open_file.fd for open_file in proc.open_files()}

                # 检查新的文件描述符
                new_fds = current_fds - prev_fds
                if new_fds:
                    with self.lock:
                        for fd in new_fds:
                            path = self._get_fd_path(fd)
                            if path:
                                self.fd_map[fd] = path
                                self.logger.info(f"新建文件描述符: {fd} -> {path}")

                prev_fds = current_fds
                time.sleep(self.interval)

            except Exception as e:
                self.logger.error(f"监控过程中发生错误: {e}")
                time.sleep(self.interval)

    def stop(self):
        """停止监控线程"""
        self.running = False
        self.join()
        self.logger.info("文件描述符监控已停止")

# 使用示例
if __name__ == "__main__":
    # 创建并启动监控线程
    monitor = FileDescriptorMonitor()
    monitor.start()

    # 测试：创建一些文件
    test_files = []
    try:
        for i in range(3):
            f = open(f"test_{i}.txt", "w")
            test_files.append(f)
            time.sleep(2)

        # 显示监控结果
        print("\n当前文件描述符映射:")
        for fd, path in monitor.get_fd_map().items():
            print(f"FD {fd}: {path}")

    finally:
        # 清理资源
        for f in test_files:
            f.close()
        monitor.stop()
import argparse
import queue
import threading
import time
import subprocess

def multiprocess_shell(commands, num_processes):
    commands = list(commands)
    task_queue = queue.Queue()
    for cmd in commands:
        task_queue.put(cmd)

    def worker(thread_id: int):
        """线程函数：不断取任务执行，执行完一个再取下一个"""
        while not task_queue.empty():
            try:
                cmd = task_queue.get_nowait()
            except queue.Empty:
                break

            print(f"[Thread-{thread_id}] Starting: {cmd}")
            process = subprocess.run(
                cmd,
                shell=True,
                # capture_output=True,  # 自动收集 stdout/stderr，防止缓冲阻塞
                # text=True,  # 自动转成 str
                check=False,  # 不抛异常，由我们自己判断 returncode
            )

            print(f"[Thread-{thread_id}] Finished: {cmd}")
            task_queue.task_done()

            # 可加短暂停顿防止打印乱序
            time.sleep(0.1)

    # 创建并启动3个线程
    threads = []
    for i in range(num_processes):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    # 等待全部线程结束
    for t in threads:
        t.join()

    print("✅ All tasks completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--commands', type=str, required=True, nargs='+', help='List of shell commands to run')
    args = parser.parse_args()
    commands = args.commands
    multiprocess_shell(commands, 4)
import sys
import time
import torch
import traceback
import gc
gc.disable()
import torch.multiprocessing as mp
import numpy as np

from threading import Thread
from queue import Empty
from torch.utils.data import IterableDataset
from typing import Any, Callable, Optional
from pympler import asizeof  # 需要安装：pip install pympler
from queue import Full

# -------------------- 模块级定义（全局作用域） --------------------
def data_generator():
    count = 0
    while count < 5:
        yield count
        count += 1
        time.sleep(0.1)


def collate_fn(x):
    return torch.tensor(x, dtype=torch.float32)


class DynamicIterableDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.count = 0  # 内部状态变量

    def __iter__(self):
        self.count = 0  # 每次调用 __iter__ 时重置状态
        while self.count < 5:
            yield self.count
            self.count += 1
            time.sleep(0.1)

    def reset(self):
        self.count = 0  # 显式重置方法

# -------------------- 消费者进程独立为类 --------------------

def move_to_cuda(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj  # 非张量或嵌套结构保持不变

class Consumer:
    def __init__(
        self,
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        stop_event,
        collate_fn: Callable,
        pin_memory: bool,
        batch_size: int,
        gpu_id: int = None,
        persistent_workers: bool = True,
    ):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.persistent_workers = persistent_workers
        
    def waiting_next_item(self):
        """持久化进程等待运行状态再次开启"""
        self.index = 0
        while True:
            try:
                with torch.cuda.nvtx.range(f"Rank{self.gpu_id}-Waiting_Next_Epoch"):
                    item = self.task_queue.get_nowait()
                break
            except Empty:
                time.sleep(0.1)
        return item

    def run(self):
        self.index = 0
        try:
            while True:
                try:
                    with torch.cuda.nvtx.range(f"Rank{self.gpu_id}-Index{self.index}-Size{self.task_queue.qsize()}-getting"):
                        item = self.task_queue.get_nowait()
                    if isinstance(item, _DestroySignal):
                        break
                    if isinstance(item, _StopSignal):
                        self.result_queue.put(_StopSignal())
                        if self.persistent_workers:
                            ret = self.waiting_next_item()
                            if isinstance(ret, _DestroySignal):
                                break
                            else:
                                item = ret
                        else:
                            break
                    with torch.cuda.nvtx.range(f"Rank{self.gpu_id}-Index{self.index}-collate_fn_qact"):
                        processed_batch = self.collate_fn(item)
                    if self.gpu_id is not None:
                        assert torch.cuda.is_available(), "CUDA is not available, but gpu_id is set."
                        with torch.cuda.nvtx.range(f"Rank{self.gpu_id}-Index{self.index}-move_to_cuda"):
                            processed_batch = move_to_cuda(processed_batch, device=f"cuda:{self.gpu_id%8}")
                    self.index += 1
                    with torch.cuda.nvtx.range(f"Rank{self.gpu_id}-Index{self.index}-putting"):
                        self.result_queue.put(processed_batch)
                except Empty:
                    time.sleep(0.1)
            while not self.stop_event.is_set():
                time.sleep(0.2)
        except Exception as e:
            print(f"[Consumer-{self.worker_id} Error] {traceback.format_exc()}", file=sys.stderr)
            self.result_queue.put(_StopSignal())
        finally:
            print(f"[Consumer-{self.worker_id} Debug] Exiting...")


# -------------------- DataLoader 实现 --------------------
class _StopSignal:
    pass

class _DestroySignal:
    pass


class DynamicDataLoader:
    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        collate_fn: Optional[Callable[[list], Any]] = None,
        pin_memory: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 1,
        gpu_id: int = None,
        length: int = None,
        persistent_workers: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn or (lambda x: x)
        self.pin_memory = pin_memory
        self.num_workers = num_workers if num_workers > 0 else 1
        self.prefetch_factor = prefetch_factor
        self.ctx = mp.get_context("spawn")
        self.gpu_id = gpu_id
        self.length = length
        self.persistent_workers = persistent_workers
        self.persistent_init = False

        self._task_queue = None
        self._result_queue = None
        self._producer_thread = None
        self._consumer_processes = None
        self._stop_event = None

    def __iter__(self):
        self._init_workers()
        return self._generator()


    def __len__(self):
        """返回DataLoader的长度"""
        if self.length is None:
            raise ValueError(
                "DynamicDataLoader's length is not set. "
                "Please set the 'length' attribute (e.g. loader.length = 100)."
            )
        return self.length

    def _init_workers(self):

        if not self.persistent_workers or not self.persistent_init:
            # 每次迭代时创建新的多进程资源
            self._stop_event = self.ctx.Event()
            self._task_queue = self.ctx.Queue(maxsize=self.num_workers * self.prefetch_factor)
            self._result_queue = self.ctx.Queue(maxsize=self.num_workers * self.prefetch_factor)
            self._consumer_processes = []

            # 启动消费者进程（仅传递必要参数，避免序列化整个 DataLoader）
            for worker_id in range(self.num_workers):
                consumer = Consumer(
                    worker_id=worker_id,
                    task_queue=self._task_queue,
                    result_queue=self._result_queue,
                    stop_event=self._stop_event,
                    collate_fn=self.collate_fn,
                    pin_memory=self.pin_memory,
                    batch_size=self.batch_size,
                    gpu_id=self.gpu_id,
                    persistent_workers=self.persistent_workers,
                )
                p = self.ctx.Process(target=consumer.run)
                p.start()
                self._consumer_processes.append(p)
            if self.persistent_workers:
                self.persistent_init = True
        else:
            # 清空队列以移除旧数据
            self._drain_queues()

        # 启动生产者线程（主线程运行）
        self._producer_thread = Thread(target=self._producer, args=(self.dataset,), daemon=True, name="DataLoader-Producer")
        self._producer_thread.start()
            
    
    def report_size(self, batch):

        def get_value_sizes(obj):
            """递归统计字典/列表中每个元素的大小，若为numpy数组则附加shape信息"""
            if isinstance(obj, dict):
                return {k: get_value_sizes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [get_value_sizes(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                # 如果是numpy数组，返回 "大小 (shape)"
                size = asizeof.asizeof(obj)
                return f"{size} bytes (shape: {obj.shape})"
            else:
                # 其他类型直接返回大小（添加 'bytes' 后缀）
                return f"{asizeof.asizeof(obj)} bytes"

        def analyze_batch(batch):
            report = []
            for idx, item in enumerate(batch):
                total_size = asizeof.asizeof(item)
                value_sizes = get_value_sizes(item)  # 获取带shape的明细
                report.append({
                    "index": idx,
                    "total_size": total_size,
                    "value_sizes": value_sizes
                })
            return report

        report = analyze_batch(batch)
        print(f"Total Dump Size: {asizeof.asizeof(batch)}")  # 单位：字节

        # 打印结果（调整输出格式）
        for entry in report:
            print(f"字典 {entry['index']} 总大小: {entry['total_size'] / 1024:.2f} KB")
            print("明细:")
            for key, size_info in entry['value_sizes'].items():
                print(f"  - {key}: {size_info}")  # size_info已包含完整信息
            print("---")


    def _producer(self, dataset: IterableDataset):
        index = 0
        try:
            batch = []
            for idx, item in enumerate(dataset):
                if self._stop_event.is_set():
                    break
                with torch.cuda.nvtx.range(f"Index{idx}-queue batch append"):
                    batch.append(item)
                with torch.cuda.nvtx.range(f"Index{idx}-queue waiting to put"):
                    if len(batch) >= self.batch_size:
                        while not self._stop_event.is_set():
                            try:
                                with torch.cuda.nvtx.range(f"Index{idx}-queue put"):
                                    self._task_queue.put(batch, block=False)
                                batch = []
                                index += 1
                                break
                            except Full:
                                time.sleep(0.1)
                        else:
                            break  # 停止事件触发，退出循环
            for _ in range(self.num_workers):
                while not self._stop_event.is_set():
                    try:
                        self._task_queue.put(_StopSignal(), block=False)
                        break
                    except Full:
                        time.sleep(0.1)
        except Exception as e:
            print(f"[Producer Error] {e}", file=sys.stderr)
            self._stop_event.set()

    def _generator(self):
        self.stop_signal = 0
        while self.stop_signal < self.num_workers:
            try:
                with torch.cuda.nvtx.range(f"iter getting result from queue"):
                    item = self._result_queue.get()
                if isinstance(item, _StopSignal):
                    self.stop_signal += 1
                    continue
                yield item
            except Empty:
                continue

    def shutdown(self, destroy=False):
        # 停止生产者
        self._stop_event.set()
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1)
        # 安全清空队列
        self._drain_queues()
        if not self.persistent_workers or destroy:
            # 等待消费者处理剩余数据
            deadline = time.time() + 5
            for p in self._consumer_processes:
                p.join(max(0, deadline - time.time()))

            # 强制终止残留进程
            for p in self._consumer_processes:
                if p.is_alive():
                    p.terminate()
                p.join()

            # 释放队列资源
            self._release_queues()

        if self.persistent_workers and not destroy:
            self._producer_thread = None
            self._stop_event.clear()

    def _drain_queues(self):
        """安全清空所有队列"""
        # 清空任务队列
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except Empty:
                break

        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Empty:
                break

    def _release_queues(self):
        """正确释放队列资源"""
        if self._task_queue is not None:
            self._task_queue.cancel_join_thread()
            self._task_queue.close()
            self._task_queue = None
        if self._result_queue is not None:
            self._result_queue.cancel_join_thread()
            self._result_queue.close()
            self._result_queue = None
        self._consumer_processes = None
        self._producer_thread = None
        self._stop_event = None

    def __del__(self):
        if self._task_queue is not None:
            print("[DynamicDataLoader] Cleaning up resources...")
            self._task_queue.put(_DestroySignal())
            if self.persistent_workers:
                self.shutdown(destroy=True)


# -------------------- 简单测试程序 --------------------
def main():
    dataset = DynamicIterableDataset()
    dataloader = DynamicDataLoader(dataset, batch_size=1, collate_fn=collate_fn, pin_memory=True, num_workers=1, prefetch_factor=1)

    for _ in range(5):
        print("Begin!")
        for batch in dataloader:
            print(f"Batch received: {batch}")
        dataloader.shutdown()
        dataset.reset()

    print("All data processed.")


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from bisect import bisect_right
import zarr
from zarr.storage import LRUStoreCache


import torch
from torch.utils.data import Dataset, Sampler
import random

class BufferedDatasetWapper(Dataset):
    def __init__(self,
        datasets:list,
        buffer_size:int):
        self.datasets = datasets
        self.buffer_size = buffer_size
        self.dataset_indices = list(range(len(datasets)))
        
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.worker_state = {}

    def cumsum(self, datasets):
        r, s = [], 0
        for dataset in datasets:
            s += len(dataset)
            r.append(s)
        return r

    def setup_data(self, state):
        start_index = state["buffer_index"] * self.buffer_size
        end_index = min(start_index + self.buffer_size, len(self.dataset_indices))
        indices = self.dataset_indices[start_index:end_index]
        
        state["indices"] = []

        # 遍历选定的缓冲区内的数据集索引
        for index in indices:
            dataset = self.datasets[index]
            store = zarr.ZipStore(dataset.zarr_path, mode='r')
            cache_size = dataset.memory_size * 1073741824 // self.buffer_size
            cached_store = LRUStoreCache(store, max_size=cache_size)
            dataset.replay_buffer = zarr.open(cached_store, mode='r')
            
            data_indices = [(index, i) for i in range(len(dataset))]
            state["indices"].extend(data_indices)

        # 对当前缓冲区内的所有数据索引进行随机排列
        state["indices"] = [state["indices"][i] for i in torch.randperm(len(state["indices"])).tolist()]
        state["local_index"] = 0

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            worker_id = 0
        else:  # in a worker process
            worker_id = worker_info.id

        # 初始化 worker 状态
        if worker_id not in self.worker_state:
            self.worker_state[worker_id] = {
                "buffer_index": 0,
                "local_index": 0,
                "indices": []
            }

        state = self.worker_state[worker_id]

        # 如果当前 buffer 的数据索引为空，则加载新的数据
        if len(state["indices"]) == 0:
            self.setup_data(state)

        dataset_index, data_index = state["indices"][index]

        # 更新状态
        state["local_index"] += 1
        if state["local_index"] >= len(state["indices"]):
            state["buffer_index"] += 1
            state["local_index"] = 0
            self.setup_data(state)
        
        dataset = self.datasets[dataset_index]
        data = dataset[data_index]

        return data

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def reset_data(self, seed=42):
        for dataset in self.datasets:
            dataset.replay_buffer = None
        self.worker_state = {}

        random.seed(seed)
        random.shuffle(self.dataset_indices)

class SlidingWindowSampler(Sampler):
    def __init__(self,
        data_source:BufferedDatasetWapper,
          buffer_size):
        self.data_source = data_source
        self.buffer_size = buffer_size

    def __iter__(self):
        shuffled_indices = self.data_source.dataset_indices
        n = len(shuffled_indices)
        buffer_indices = [shuffled_indices[i:i+self.buffer_size] for i in range(0, n, self.buffer_size)]

        return iter([i for sublist in buffer_indices for item in sublist for i in range(len(self.data_source.datasets[item]))])

    def __len__(self):
        return len(self.data_source)
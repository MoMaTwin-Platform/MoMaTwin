import torch
from torch.utils.data import Dataset, Sampler

import random
import numpy as np
import copy
import time

import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # 以MB为单位
    return mem

import inspect

class BufferedDatasetWapper(Dataset):
    def __init__(self,
        datasets:list,
        buffer_size:int,
        seed:int=42,):

        super().__init__()

        self.datasets = datasets
        self.buffer_size = buffer_size
        self.dataset_indices = list(range(len(datasets)))
        self.sizes = self.lengths(datasets)

        self.data_indices = None
        self.buffer_indices = None
        self.worker_state = {}

        self.seed = seed
        self.reset_data(0)

    def _get_buffer(self):
        group_indices = [i // self.buffer_size for i in range(len(self.dataset_indices))]
        buffer_indices = []
        for group_idx, dataset_idx in zip(group_indices, self.dataset_indices):
            buffer_indices.extend([group_idx]*self.sizes[dataset_idx])

        return buffer_indices

    def lengths(self, datasets):
        s = []
        for dataset in datasets:
            s.append(len(dataset))
        return s

    def setup_data(self, worker_id, state):
        #unload_data from previous buffer
        #if worker_id == 0:
        #    print(f"worker {worker_id}: buffer {state['buffer_index']} {os.path.basename(__file__)} at  {inspect.currentframe().f_lineno} Memory usage: {memory_usage()}")
        datasets = state["datasets"]
        for dataset in datasets:
            if hasattr(dataset, 'unload_data'):
                dataset.unload_data()

        #if worker_id == 0:
        #    print(f"worker {worker_id}: buffer {state['buffer_index']}  {os.path.basename(__file__)} at  {inspect.currentframe().f_lineno} Memory usage: {memory_usage()}")

        start_index = state["buffer_index"] * self.buffer_size
        end_index = min(start_index + self.buffer_size, len(self.dataset_indices))
        indices = self.dataset_indices[start_index:end_index]        

        for index in indices:
            dataset = datasets[index]
            if hasattr(dataset, "load_data"):
                dataset.load_data(buffer_size=self.buffer_size)
        #if worker_id == 0:
        #    print(f"worker {worker_id}: buffer {state['buffer_index']}  {os.path.basename(__file__)} at  {inspect.currentframe().f_lineno} Memory usage: {memory_usage()}")
        

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            worker_id = 0
        else:
            worker_id = worker_info.id

        if worker_id not in self.worker_state:
            for dataset in self.datasets:
                if hasattr(dataset, 'unload_data'):
                    dataset.unload_data()
            self.worker_state[worker_id] = {
                "buffer_index": -1,
                # "datasets": copy.deepcopy(self.datasets) 
                "datasets": self.datasets 
            }

        state = self.worker_state[worker_id]

        if self.buffer_indices[index] != state['buffer_index']:
            state['buffer_index'] = self.buffer_indices[index]
            self.setup_data(worker_id, state)

        dataset_idx, data_idx = self.data_indices[index]
        dataset = state["datasets"][dataset_idx]
        
        # return dataset[data_idx]
        try:
            return dataset[data_idx]
        except:
            print(f"Error occurred at {os.path.basename(__file__)} at  {inspect.currentframe().f_lineno}")
            return None


    def __len__(self):
        return sum(self.sizes)
    
    def reset_data(self, epoch_id):
        self.worker_state = {}

        random.seed(self.seed + epoch_id)
        random.shuffle(self.dataset_indices)

        data_indices = []
        for index in self.dataset_indices:
            dataset = self.datasets[index]
            data_indices.extend([(index, i) for i in range(self.sizes[index])])

        self.data_indices = data_indices
        self.buffer_indices = self._get_buffer()

        for dataset in self.datasets:
            if hasattr(dataset, 'shuffle_by_seed'):
                dataset.shuffle_by_seed(self.seed+epoch_id)
            if hasattr(dataset, 'unload_data'):
                dataset.unload_data()
        
        self.worker_state = {}

class SlidingWindowSampler(Sampler):
    def __init__(self,
        data_source:BufferedDatasetWapper,
        dataset_weights=None,
        replacement=False,
        ):
        super().__init__(data_source)
        self.data_source = data_source
        self.buffer_size = data_source.buffer_size
        self.seed = data_source.seed

        self.dataset_weights = dataset_weights
        self.replacement = replacement

    def __iter__(self):
        shuffled_indices = self.data_source.dataset_indices
        buffer_indices = [shuffled_indices[i:i+self.buffer_size] for i in range(0, len(shuffled_indices), self.buffer_size)]

        local_shuffled_sum = [sum(self.data_source.sizes[item] for item in sublist) for sublist in buffer_indices]
        ends = [0] + np.cumsum(local_shuffled_sum).tolist()
        starts, ends = ends[:-1], ends[1:] # len: len(data_source)//buffer_size

        random.seed(self.seed)
        if self.dataset_weights is None:
            local_shuffled_indices = []
            for start,end in zip(starts, ends):
                local_indices = list(range(start, end))
                random.shuffle(local_indices)
                local_shuffled_indices.extend(local_indices)
        
            return iter(local_shuffled_indices)
        
        assert len(self.dataset_weights) == len(self.data_source.datasets), \
                f"dataset_weights (len={len(self.dataset_weights)}) should have the same length as the number of datasets (len={len(self.data_source.datasets)})"
        
        local_shuffled_indices = []
        for i in range(len(starts)):
            buffer_dataset_indices = shuffled_indices[i*self.buffer_size:(i+1)*self.buffer_size]
            buffer_dataset_sizes = [self.data_source.sizes[idx] for idx in buffer_dataset_indices]

            start = starts[i]
            buffer_indices = []
            for j in range(len(buffer_dataset_sizes)):
                size = buffer_dataset_sizes[j]
                dataset_idx = buffer_dataset_indices[j]
                num_samples = int(size*self.dataset_weights[dataset_idx])
                end = start + size
                local_dataset_indices = list(range(start, end))
                sampled_indices = np.random.choice(local_dataset_indices, size=num_samples, replace=self.replacement)
                buffer_indices.extend(sampled_indices.tolist())
                start = end
            random.shuffle(buffer_indices)
            local_shuffled_indices.extend(buffer_indices)
    
        return iter(local_shuffled_indices)

        

    def __len__(self):
        return len(self.data_source)

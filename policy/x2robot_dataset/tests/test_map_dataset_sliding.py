import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from bisect import bisect_right
import zarr
from zarr.storage import LRUStoreCache


import torch
from torch.utils.data import Dataset
# import zarr
# from numcodecs import LRUStoreCache
import random

class OptimizedDataset(Dataset):
    def __init__(self, datasets, buffer_size):
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

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
import torch.distributed as dist

from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
import os
import torch
import tqdm

from x2robot_dataset.common.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING
)
from PIL import Image

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16155'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

import json

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from bisect import bisect_right
import zarr
from zarr.storage import LRUStoreCache
import numpy as np


from torch.utils.data import Sampler
class SlidingWindowSampler(Sampler):
    def __init__(self, data_source, buffer_size):
        self.data_source = data_source
        self.buffer_size = buffer_size

    def __iter__(self):
        shuffled_indices = self.data_source.dataset_indices
        n = len(shuffled_indices)
        buffer_indices = [shuffled_indices[i:i+self.buffer_size] for i in range(0, n, self.buffer_size)]

        return iter([i for sublist in buffer_indices for item in sublist for i in range(len(self.data_source.datasets[item]))])

    def __len__(self):
        return len(self.data_source)

import tqdm
def main(rank, world_size):
    setup(rank, world_size)

    data_mx_file = '/x2robot/liangyuxin/workspace/DiffusionPolicy/big_mix_0617_mn/all_data.json'
    with open(data_mx_file, 'r') as f:
        data = json.load(f)
    dataset_paths = [task['path'] for task in data]

    dataset_paths = dataset_paths[3:16]

    default_data_configs = {'default_instruction': 'spread the clothes',
                            'action_horizon': 100,
                            'action_history_length': 0,
                            'image_horizon': 1,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': False,
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            'obs_keys': list(_CAM_FILE_MAPPING.keys()),
                            'action_keys': list(_ACTION_KEY_EE_MAPPING.keys())
                            }

    global train_dataset
    global val_dataset
    train_dataset, val_dataset, _, _ = make_chunk_dataset(
                       dataset_paths,
                       rank=rank,
                       dp_world_size=world_size,
                       cache_dir='/x2robot/Data/.cache', #缓存数据集的地址，可以被其它进程共享
                       read_labeled_data=False, #是否读取人工标注的数据
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=16, #内存使用大小, GB
                       return_full_traj=False, #是否返回完整轨迹, default False
                       flush_cache=False, #是否清空缓存
                       data_configs=[default_data_configs] * len(dataset_paths), #数据配置
                       num_workers=20, #数据处理的进程数
                       filter_angle_outliers=True, #是否过滤角度异常值
                       detect_motion=True, #是否去掉静止不动的样本
                       trim_stationary=False) #是否去掉首尾不动的部分

    collate_fn = collate_wrapper(obs_keys=default_data_configs['obs_keys'],
                                 collate_type = 'chunking', #chuning, full
                                 low_dim_obs_horizon=1,
                                 img_obs_horizon=1,
                                 horizon=20,
                                 action_dim=14,
                                 is_bi_mode=True)


    train_dataset = OptimizedDataset(train_dataset.datasets, buffer_size=4)
    train_sampler = SlidingWindowSampler(train_dataset, buffer_size=4)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=20, collate_fn=collate_fn)

    num_epochs = 3
    for epoch in range(num_epochs):
        train_dataset.reset_data(seed=42)
        cc = 0
        for batch_data in tqdm.tqdm(train_loader):
            batch = batch_data
            print(f'rank {rank} epoch {epoch} batch {cc} frame {batch["frame"]} uid {batch["uid"]}', flush=True)
            # print(batch['dataset_name'], batch['frame'], batch['uid'])
            # if cc == 20:
            #     break
            # cc += 1

if __name__ == '__main__':
    world_size = 1
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)


    

import torch
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
from x2robot_dataset.load_datasets import (
    get_data_config_list,
    make_chunk_dataset_from_config,
    DataGroupBatchSampler,
)
import os
import torch
import tqdm
import random

from x2robot_dataset.common.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING
)

import torch.distributed as dist
from accelerate import Accelerator

import tqdm
import json
import functools

global train_dataset
train_dataset = None
def init_worker_train(worker_id):
    global train_dataset
    if train_dataset is not None:
        for dataset in train_dataset.datasets:
            if hasattr(dataset, "load_data"):
                dataset.load_data()

global val_dataset
val_dataset = None
def init_worker_val(worker_id):
    global val_dataset
    if val_dataset is not None:
        for dataset in val_dataset.datasets:
            if hasattr(dataset, "load_data"):
                dataset.load_data()

class TestModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16275'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main():
    rank = 0
    world_size = 1
    setup(rank, world_size)
    data_mx_file = '/x2robot/liangyuxin/workspace/DiffusionPolicy/big_mix_0617_mn/all_data.json'
    with open(data_mx_file, 'r') as f:
        data = json.load(f)

    print(f'rank {rank} world_size {world_size}', flush=True)

    default_data_configs = [{"path": d["path"],
                            'default_instruction': d["instruction"],
                            'action_horizon': 21,
                            'action_history_length': 20,
                            'image_horizon': 1,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': True,
                            'obs_keys': ["face_view", "left_wrist_view", "right_wrist_view"],
                            'action_keys': list(_ACTION_KEY_EE_MAPPING.keys()),
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            } for d in data] 
        
    # 先准备好所有的data config
    dataload_config, dataload_args = get_data_config_list(
                    data_configs=default_data_configs, 
                    datachunk_size=8, #每个数据块的大小,一般可以32~64之间
                    rank=rank,
                    dp_world_size=world_size,
                    read_labeled_data=False, #是否读取人工标注的数据
                    trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                    return_full_traj=False, #是否返回完整轨迹, default False
                    flush_cache=False, #是否清空缓存
                    num_workers=20, #数据处理的进程数
                    filter_angle_outliers=True, #是否过滤角度异常值
                    detect_motion=True, #是否去掉静止不动的样本
                    trim_stationary=False, #是否去掉首尾不动的部分
                    ) 

        
    print(f"len of dataload_config: {len(dataload_config)}",flush=True)
    for i in range(len(dataload_config)):
        print(f"dataload_config[{i}]: {dataload_config[i]}",flush=True)
    print("dataload_args", dataload_args)

    num_epochs = 3
    for epoch in range(num_epochs):
        for i in range(len(dataload_config)):
            print(f"rank {rank} epoch {epoch} {i}/{len(dataload_config)}",flush=True)
            global train_dataset
            global val_dataset

            # 每次加载一部分数据
            train_dataset, val_dataset, train_weight, val_weight = make_chunk_dataset_from_config(dataload_config[i],dataload_args=dataload_args)
            batch_size = 8
            train_sampler = DataGroupBatchSampler(train_dataset,batch_size)
            val_sampler = DataGroupBatchSampler(val_dataset,batch_size)
            collate_fn = functools.partial(collate_wrapper(),
                            low_dim_obs_horizon=1,
                            img_obs_horizon=1,
                            horizon=16,
                            sample2instruct=None)

            val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=8, worker_init_fn=init_worker_val)
            train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=8, worker_init_fn=init_worker_train)
            for batch_data in tqdm.tqdm(train_dataloader, desc=f'epoch {epoch}'):
                print(batch_data.keys())
                print(batch_data['obs'].keys())
                break
            break
        break    

if __name__ == '__main__':
    main()


    

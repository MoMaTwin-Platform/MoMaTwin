import torch
from torch.utils.data import DataLoader

import torch
from torch.utils.data import Dataset

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

import torch.distributed as dist
from accelerate import Accelerator

import tqdm
import json


def main():
    accelerator = Accelerator()

    rank = accelerator.state.process_index
    world_size = accelerator.state.num_processes

    # data_mx_file = '/x2robot/liangyuxin/workspace/DiffusionPolicy/fold_clothes_0625_from_mix_mn/clothes_task.json'
    data_mx_file = '/x2robot/liangyuxin/workspace/DiffusionPolicy/big_mix_0617_mn/all_data.json'
    with open(data_mx_file, 'r') as f:
        data = json.load(f)
    dataset_paths = [task['path'] for task in data]

    dataset_paths = dataset_paths[:10]
    print('dataset_paths ', dataset_paths)

    # dataset_paths = dataset_paths[:10]
    #dataset_paths = ['/x2robot/zhengwei/10001/20240530-hang-clothes',\
    #            '/x2robot/zhengwei/10001/20240530-hang-clothes-addition',\
    #            '/x2robot/zhengwei/10001/20240531-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240531-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240531-hang-clothes-addition',\
    #            '/x2robot/zhengwei/10002/20240531-hang-clothes-second',\
    #            '/x2robot/zhengwei/10002/20240604-hang-clothes',\
    #            '/x2robot/zhengwei/10001/20240604-hang-clothes',\
    #            '/x2robot/zhengwei/10001/20240604-hang-clothes-addition',\
    #            '/x2robot/zhengwei/10001/20240605-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240605-hang-clothes',\
    #            '/x2robot/zhengwei/10001/20240611-hang-clothes',\
    #            '/x2robot/zhengwei/10001/20240611-hang-clothes-addtion',\
    #            '/x2robot/zhengwei/10002/20240611-hang-clothes',\
    #            '/x2robot/zhengwei/10000/20240613-hang-clothes',\
    #            '/x2robot/zhengwei/10000/20240614-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240614-hang-clothes-rlhf',\
    #            '/x2robot/zhengwei/10001/20240617-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240618-hang-clothes-rlhf',\
    #            '/x2robot/zhengwei/10001/20240618-hang-clothes-rlhf',\
    #            '/x2robot/zhengwei/10001/20240621-hang-clothes--1',\
    #            '/x2robot/zhengwei/10001/20240621-hang-clothes-night',\
    #            '/x2robot/zhengwei/10001/20240622-hang-clothes',\
    #            '/x2robot/zhengwei/10002/20240626-hang-clothes--3',\
    #            '/x2robot/zhengwei/10001/20240626-hang-clothes']

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

    train_dataset, val_dataset, train_sampler, val_sampler = make_chunk_dataset(
                       dataset_paths,
                    #    sample_ratio=[0.2]*len(dataset_paths),
                       rank=rank,
                       dp_world_size=world_size,
                       dataset_buffer_size=3, #数据集缓冲区大小
                       cache_dir='/x2robot/Data/.cache', #缓存数据集的地址，可以被其它进程共享
                       read_labeled_data=False, #是否读取人工标注的数据
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=4, #内存使用大小, GB
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
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=20, collate_fn=collate_fn)


    train_loader = accelerator.prepare(train_loader)
    num_epochs = 100
    for epoch in range(num_epochs):
        train_dataset.reset_data(epoch_id=epoch)
        # cc = 0
        for batch_data in tqdm.tqdm(train_loader, desc=f'epoch {epoch}'):
            batch = batch_data
            #print(f'rank {rank} epoch {epoch} batch {cc} frame {batch["frame"]} uid {batch["uid"]}', flush=True)
    
    accelerator.wait_for_everyone()
if __name__ == '__main__':
    main()


    

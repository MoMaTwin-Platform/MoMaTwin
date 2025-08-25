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
    _CAM_FILE_MAPPING,
    _TAC_FILE_MAPPING
)
from PIL import Image

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '15155'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

import matplotlib.pyplot as plt

global train_dataset
train_dataset = None
def init_worker(worker_id):
    global train_dataset
    if train_dataset is not None:
        for dataset in train_dataset.datasets:
            dataset.load_data()

global val_dataset
val_dataset = None
def init_worker(worker_id):
    global val_dataset
    if val_dataset is not None:
        for dataset in val_dataset.datasets:
            dataset.load_data()

def main(rank, world_size):
    setup(rank, world_size)

    dataset_paths = ['/x2robot/zhengwei/10003/20240510-touch']

    default_data_configs = {'default_instruction': 'pick up the egg and put it down aside',
                            'action_horizon': 100,
                            'action_history_length': 0,
                            'image_horizon': 2,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': False,
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            'tac_keys': list(_TAC_FILE_MAPPING.values()),
                            'obs_keys': ['right_wrist_view'],
                            'action_keys': ['follow_right_ee_cartesian_pos',
                                            'follow_right_ee_rotation',
                                            'follow_right_gripper']
                            }

    global train_dataset
    global val_dataset
    train_dataset, val_dataset, _, _ = make_chunk_dataset(
                       dataset_paths,
                       rank=rank,
                       dp_world_size=world_size,
                       cam_mapping={'rightImg':'right_wrist_view'},
                       cache_dir='/x2robot/Data/.cache', #缓存数据集的地址，可以被其它进程共享
                       read_labeled_data=True, #是否读取人工标注的数据
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=4, #内存使用大小, GB
                       parse_tactile=True, #是否解析触觉数据
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
                                 img_obs_horizon=2,
                                 horizon=20,
                                 action_dim=14,
                                 is_bi_mode=False)


    for dataset in train_dataset.datasets:
        dataset.shuffle_by_seed(42)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(train_dataset,
                             batch_size=10,
                             sampler=sampler,
                             num_workers=12,
                             collate_fn=collate_fn,
                             worker_init_fn=init_worker)
    cc = 0
    for batch in tqdm.tqdm(data_loader):
        print(batch['tactiles']['left_tactile'].shape)
        if cc == 50:
            break
        cc += 1

if __name__ == '__main__':
    world_size = 1
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)

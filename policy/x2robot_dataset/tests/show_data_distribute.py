from x2robot_dataset.map_dataset import make_chunk_dataset, collate_wrapper
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import argparse
from typing import List

from x2robot_dataset.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING,
    _CAM_BINOCULAR_FILE_MAPPING,
    _CAM_MAPPING,
    _HEAD_ACTION_MAPPING,
    _CAM_BINOCULAR_MAPPING,
)
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description='Action Statistics Analysis')
    parser.add_argument('--dataset_paths', nargs='+',
                        help='List of dataset paths')
    parser.add_argument('--parse_head_action', action='store_true',
                        help='Whether to parse head action')
    parser.add_argument('--is_binocular', action='store_true',
                        help='Whether to parse binocular view of head')
    parser.add_argument('--output_dir', type=str, default='action_statistics',
                        help='Directory to save output files')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='File containing dataset paths (one path per line)')
    return parser.parse_args()

def read_dataset_paths(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        # if line start with #, it is a comment
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

def plot_action_distribution(actions, dim, save_dir, action_names):
    """Plot and save distribution for each action dimension"""
    plt.figure(figsize=(10, 6))
    plt.hist(actions.flatten(), bins=50, density=True)
    plt.title(f'Distribution of {action_names[dim]}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(save_dir, f'action_dist_{dim}.png'))
    plt.close()

def main():
    args = parse_args()
    
    # 处理数据集路径
    if args.dataset_file:
        dataset_paths = read_dataset_paths(args.dataset_file)
    else:
        dataset_paths = args.dataset_paths
    print(f'Dataset Paths: {dataset_paths}')

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    print(f'{rank}/{world_size}')

    # 创建保存图像的目录
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # 设置动作相关的配置
    parse_head_action = args.parse_head_action
    is_binocular = args.is_binocular
    action_keys = (list(_ACTION_KEY_EE_MAPPING.keys()) + 
                  list(_HEAD_ACTION_MAPPING.keys()) if parse_head_action 
                  else list(_ACTION_KEY_EE_MAPPING.keys()))
    
    action_names = (['lx','ly','lz','lx_rot', 'ly_rot', 'lz_rot', 'l_gripper', 
                    'rx','ry','rz','rx_rot', 'ry_rot', 'rz_rot', 'r_gripper', 
                    'head_yaw', 'head_pitch'] if parse_head_action 
                   else ['lx','ly','lz','lx_rot', 'ly_rot', 'lz_rot', 'l_gripper',
                         'rx','ry','rz','rx_rot', 'ry_rot', 'rz_rot', 'r_gripper'])
    
    action_dim = 16 if parse_head_action else 14

    action_history_length = 0
    merge_cur_history = action_history_length > 0
    default_data_configs = {
        'default_instruction': 'pick up item onto container.',
        'action_horizon': 21,
        'action_history_length': action_history_length,
        'image_horizon': 1,
        'image_history_length': 0,
        'right_padding': True,
        'left_padding': True,
        'train_val_split': 0.9,
        'split_seed': 42,
        'obs_keys': list(_CAM_FILE_MAPPING.keys()) if not is_binocular else list(_CAM_BINOCULAR_FILE_MAPPING.keys()),
        'action_keys': action_keys
    }

    train_dataset, val_dataset, train_sampler, _ = make_chunk_dataset(
        dataset_paths,
        rank=rank,
        dp_world_size=world_size,
        cam_mapping=_CAM_BINOCULAR_MAPPING if is_binocular else _CAM_MAPPING,
        cache_dir='/x2robot/Data/.cache',
        read_labeled_data=True,
        dataset_buffer_size=20,
        trans_zh2en_url=None,
        read_from_cache=False,
        memory_size=8,
        parse_tactile=False,
        parse_head_action=parse_head_action,
        return_full_traj=False,
        flush_cache=False,
        data_configs=[default_data_configs] * len(dataset_paths),
        num_workers=16,
        filter_angle_outliers=True,
        detect_motion=True,
        trim_stationary=False)

    collate_fn = collate_wrapper(
        obs_keys=default_data_configs['obs_keys'],
        low_dim_obs_horizon=1,
        img_obs_horizon=1,
        horizon=20,
        action_dim=action_dim,
        is_bi_mode=True,
        parse_head_action=parse_head_action,
        merge_cur_history=merge_cur_history
    )

    for dataset in train_dataset.datasets:
        dataset.shuffle_by_seed(42)

    data_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=40,
        collate_fn=collate_fn
    )
    data_loader = accelerator.prepare(data_loader)
    print(f'len:{len(data_loader)}', flush=True)

    # 初始化统计数据的数组
    action_stats = {
        'min': np.ones(action_dim) * float('inf'),
        'max': np.ones(action_dim) * float('-inf'),
        'mean': np.zeros(action_dim),
        'std': np.zeros(action_dim),
        'all_values': [[] for _ in range(action_dim)]
    }

    # 收集统计数据
    total_samples = 0
    for batch in tqdm.tqdm(data_loader):
        actions = batch['action'].cpu().numpy()
        batch_size = actions.shape[0]
        total_samples += batch_size

        # 更新统计数据
        for dim in range(action_dim):
            action_dim_data = actions[:, :, dim]
            action_stats['min'][dim] = min(action_stats['min'][dim], np.min(action_dim_data))
            action_stats['max'][dim] = max(action_stats['max'][dim], np.max(action_dim_data))
            action_stats['all_values'][dim].extend(action_dim_data.flatten())

    # 计算均值和标准差
    for dim in range(action_dim):
        action_stats['mean'][dim] = np.mean(action_stats['all_values'][dim])
        action_stats['std'][dim] = np.std(action_stats['all_values'][dim])

    # 打印统计结果
    if rank == 0:
        # 保存统计结果到文本文件
        stats_file = os.path.join(args.output_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            # save dataset paths
            print("\nDataset Paths:", file=f)
            for path in dataset_paths:
                print(path, file=f)

            print("\nAction Statistics:", file=f)
            for dim in range(action_dim):
                print(f"\nDimension {dim} ({action_names[dim]}):", file=f)
                print(f"Min: {action_stats['min'][dim]:.4f}", file=f)
                print(f"Max: {action_stats['max'][dim]:.4f}", file=f)
                print(f"Mean: {action_stats['mean'][dim]:.4f}", file=f)
                print(f"Std: {action_stats['std'][dim]:.4f}", file=f)
                
                # 绘制并保存分布图
                plot_action_distribution(
                    np.array(action_stats['all_values'][dim]), 
                    dim, 
                    args.output_dir,
                    action_names
                )

    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
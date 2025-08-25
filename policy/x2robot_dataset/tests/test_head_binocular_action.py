from torch.utils.data import DataLoader
from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
import tqdm
import numpy as np

from x2robot_dataset.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING,
    _CAM_BINOCULAR_FILE_MAPPING,
    _CAM_MAPPING,
    _CAM_BINOCULAR_MAPPING,
    _HEAD_ACTION_MAPPING,
)
from accelerate import Accelerator
def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size=accelerator.num_processes
    print(f'{rank}/{world_size}')
    # dataset_paths = ['/x2robot/zhengwei/10005/20241016-pick_up-item','/x2robot/zhengwei/10005/20241017-pick_up-item-leju','/x2robot/zhengwei/10005/20241017-pick_up-item-leju', '/x2robot/zhengwei/10005/20241018-pick_up-item-leju']
    dataset_paths = [
        # '/x2robot/zhengwei/10005/20241023-pick_up-item-leju',
        # '/x2robot/zhengwei/10005/20241024-pick_up-item-leju',
        # '/x2robot/zhengwei/10005/20241024-pick_up-item-leju-1',
        # '/x2robot/zhengwei/10005/20241028-pick_up-item-leju',
        # '/x2robot/zhengwei/10005/20241028-pick_up-item-leju-night',
        '/x2robot/zhengwei/10005/20241029-pick_up-item-leju'
    ]
    action_history_length = 0
    merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
    default_data_configs = {'default_instruction': 'pick up item onto container.',
                            'action_horizon': 100,
                            'action_history_length': action_history_length,
                            'image_horizon': 1,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': True,
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            'obs_keys': list(_CAM_BINOCULAR_FILE_MAPPING.keys()),
                            'action_keys': list(_ACTION_KEY_EE_MAPPING.keys())+list(_HEAD_ACTION_MAPPING.keys()) # must set with parse_head_action=True below for adding head action
                            }

    train_dataset, val_dataset, train_sampler, _ = make_chunk_dataset(
                       dataset_paths,
                       rank=rank,
                       dp_world_size=world_size,
                       cache_dir='/x2robot/Data/.cache', #缓存数据集的地址，可以被其它进程共享
                       cam_mapping=_CAM_BINOCULAR_MAPPING,
                       dataset_buffer_size=20,
                       read_labeled_data=True, #是否读取人工标注的数据
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=8, #内存使用大小, GB
                       parse_tactile=False, #是否解析触觉数据
                       parse_head_action=True, #是否解析头部数据
                       return_full_traj=False, #是否返回完整轨迹, default False
                       flush_cache=False, #是否清空缓存
                       data_configs=[default_data_configs] * len(dataset_paths), #数据配置
                       num_workers=16, #数据处理的进程数
                       filter_angle_outliers=True, #是否过滤角度异常值
                       detect_motion=True, #是否去掉静止不动的样本
                       trim_stationary=False) #是否去掉首尾不动的部分

    collate_fn = collate_wrapper(obs_keys=default_data_configs['obs_keys'],
                                 low_dim_obs_horizon=1,
                                 img_obs_horizon=1,
                                 horizon=20,
                                 action_dim=16, # add head_action [2]
                                 is_bi_mode=True,
                                 parse_head_action=True,
                                 merge_cur_history=merge_cur_history)


    # for dataset in train_dataset.datasets:
    #     dataset.shuffle_by_seed(42)

    data_loader = DataLoader(train_dataset,
                             batch_size=64,
                             sampler=train_sampler,
                             num_workers=20,
                             collate_fn=collate_fn)
    data_loader = accelerator.prepare(data_loader)
    print(f'len:{len(data_loader)}', flush=True)
    cc = 0
    min_values = []
    max_values = []
    mean_values = []

    for batch_idx, batch in enumerate(data_loader):
        b = batch['action'].cpu().numpy()
        
        # 计算最后一个维度的最小值、最大值和均值
        min_last_dim = np.min(b, axis=(0, 1))
        max_last_dim = np.max(b, axis=(0, 1))
        mean_last_dim = np.mean(b, axis=(0, 1))
        
        min_values.append(min_last_dim)
        max_values.append(max_last_dim)
        mean_values.append(mean_last_dim)

    # 在所有批次上计算全局的最小值、最大值和均值
    global_min = np.min(min_values, axis=0)
    global_max = np.max(max_values, axis=0)
    global_mean = np.mean(mean_values, axis=0)

    # 确保所有进程都完成了计算
    accelerator.wait_for_everyone()
    names = 'lx,ly,lz,lrx,lry,lrz,lg,rx,ry,rz,rrx,rry,rrz,rg,hyaw,hpitch'.split(',')
    index2name = {i:names[i] for i in range(16)}
    # 只在主进程上打印结果
    if accelerator.is_main_process:
        for i in range(len(global_min)):
            print(f'Dimension {index2name[i]}:')
            print(f'  Min: {global_min[i]}')
            print(f'  Max: {global_max[i]}')
            print(f'  Mean: {global_mean[i]}') 
if __name__ == '__main__':
    main()
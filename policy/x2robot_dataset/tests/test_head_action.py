from torch.utils.data import DataLoader
from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
import tqdm

from x2robot_dataset.common.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING,
    _HEAD_ACTION_MAPPING,
)
from accelerate import Accelerator
def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size=accelerator.num_processes
    print(f'{rank}/{world_size}')
    # dataset_paths = ['/x2robot/zhengwei/10004/20240928-brush-toilet-VR']
    dataset_paths = ['/x2robot/zhengwei/10004/20241018-brush-toilet-VR',
                     '/x2robot/zhengwei/10004/20241018-brush-toilet-VR-night',
                    '/x2robot/zhengwei/10004/20241021-brush-toilet-VR',
                    '/x2robot/zhengwei/10004/20241021-brush-toilet-VR-night-1',
                    '/x2robot/zhengwei/10004/20241022-brush-toilet',
                    '/x2robot/zhengwei/10004/20241022-brush-toilet-VR']
    dataset_paths = ['/x2robot/zhengwei/10004/20241023-brush-toilet']
    action_history_length = 0
    merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
    default_data_configs = {'default_instruction': 'pick up item onto container.',
                            'action_horizon': 21,
                            'action_history_length': action_history_length,
                            'image_horizon': 1,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': True,
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            'obs_keys': list(_CAM_FILE_MAPPING.keys()),
                            'action_keys': list(_ACTION_KEY_EE_MAPPING.keys())+list(_HEAD_ACTION_MAPPING.keys()) # must set with parse_head_action=True below for adding head action
                            }

    train_dataset, val_dataset, train_sampler, _ = make_chunk_dataset(
                       dataset_paths,
                       rank=rank,
                       dp_world_size=world_size,
                       cache_dir='/x2robot/Data/.cache', #缓存数据集的地址，可以被其它进程共享
                       read_labeled_data=True, #是否读取人工标注的数据
                       dataset_buffer_size=20,
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=4, #内存使用大小, GB
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


    for dataset in train_dataset.datasets:
        dataset.shuffle_by_seed(42)

    data_loader = DataLoader(train_dataset,
                             batch_size=64,
                             sampler=train_sampler,
                             num_workers=4,
                             collate_fn=collate_fn)
    data_loader = accelerator.prepare(data_loader)
    print(f'len:{len(data_loader)}', flush=True)
    cc = 0
    import numpy as np
    min_yaw = 1000
    max_yaw = -1000
    min_pitch = 1000
    max_pitch = -1000
    # for batch in tqdm.tqdm(data_loader):
    for batch_idx, batch in enumerate(data_loader): 
        b = batch['action'].cpu().numpy()
        miny = np.min(b[:,:,14])
        maxy = np.max(b[:,:,14])
        min_yaw = min(min_yaw, miny)
        max_yaw = max(max_yaw, maxy)

        minp = np.min(b[:,:,15])
        maxp = np.max(b[:,:,15])
        min_pitch = min(min_pitch, minp)
        max_pitch = max(max_pitch, maxp)

        # print(batch['action'].shape)
        # print(batch['action_history'].shape)
        # if cc == 1:
        #     print(batch['action'])
        #     break
        cc += 1
        break
    print('yaw: ', min_yaw, max_yaw)
    print('pitch: ', min_pitch, max_pitch)
    
    accelerator.wait_for_everyone()
if __name__ == '__main__':
    main()
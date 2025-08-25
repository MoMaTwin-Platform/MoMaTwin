import torch
from filelock import FileLock
import shutil
import pathlib
from typing import Dict
import random
from collections import defaultdict

from x2robot_dataset.common.replay_buffer import ReplayBuffer
from x2robot_dataset.common.lie_group import (
    so3_to_lie_algebra,
    euler_to_rotation_matrix
)
from x2robot_dataset.common.mask import (
    overlay_masks,
    overlay_mask_on_image,
    overlay_box_on_image
)

from x2robot_dataset.common.data_utils import (
    split_dataset_with_list,
    print_rank0
)
from x2robot_dataset.common.data_preprocessing import (
    LabelData,
    MaskMetaData,
    _ACTION_KEY_EE_MAPPING,
    _HEAD_ACTION_MAPPING,
    _CAM_FILE_MAPPING,
    _CAM_MAPPING
)
import zarr
import numpy as np
import os
import glob

from x2robot_dataset.replay_buffer import X2ReplayBuffer
from x2robot_dataset.data_buffer import BufferedDatasetWapper, SlidingWindowSampler

from typing import List, Dict
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import Sampler 
import functools
from zarr.storage import LRUStoreCache
from zarr.sync import ThreadSynchronizer
import tqdm
import gc
import psutil
import inspect

def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # 以MB为单位
    return mem

MASK2VIEW_DICT = {
    'face_mask': 'face_view',
    'left_mask': 'left_wrist_view',
    'right_mask': 'right_wrist_view',
}
VIEW2MASK_DICT = {v:k for k,v in MASK2VIEW_DICT.items()}

class MapDataset(torch.utils.data.Dataset):
    def __init__(self,
                 rank=0,
                 lock_path:str=None,
                 cache_path:str=None,
                 zarr_path:str=None,
                 read_from_cache:bool=True,
                 flush_cache:bool=False,
                 memory_size:int=8,#GB
                 action_sampling_rate=1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # with FileLock(lock_path):

        action_sampling_step = int(max(1/min(max(action_sampling_rate, 0.05), 1), 1)) #假定频率20hz

        replay_buffer = None
        if read_from_cache:
            if rank == 0:
                # cache does not exist
                if not os.path.exists(cache_path) or flush_cache:
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=True
                            ) as lmdb_store:
                            with zarr.ZipStore(zarr_path, mode='r') as zip_store:
                                print_rank0(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print_rank0("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            torch.distributed.barrier()
        
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )

        self.zarr_path = zarr_path
        self.replay_buffer = replay_buffer
        self.action_sampling_step = action_sampling_step

        self.memory_size = memory_size
        self.available_indices = None
        self.starts,self.ends = None,None
        self.sample_names = None

    def load_data(self, buffer_size=1):
        self.store = zarr.ZipStore(self.zarr_path, mode='r')
        self.cached_store = self.store
        self.replay_buffer = zarr.open(self.cached_store, mode='r',synchronizer=ThreadSynchronizer())
        # try:
        #     self.store = zarr.ZipStore(self.zarr_path, mode='r')
        #     self.cached_store = self.store
        #     self.replay_buffer = zarr.open(self.cached_store, mode='r',synchronizer=ThreadSynchronizer())
        #     # cache_size = self.memory_size * 1073741824 // buffer_size
        #     # self.cached_store = LRUStoreCache(self.store, max_size=cache_size)
        #     # self.replay_buffer = zarr.open(self.cached_store, mode='r',synchronizer=ThreadSynchronizer())        
        # except:
        #     print_rank0(f'zarr_path:{self.zarr_path}', flush=True)
    
    def unload_data(self):
        try:
            self.store.close()
            self.cached_store.clear()
        except:
            pass
        self.store = None
        self.cached_store = None
        self.replay_buffer = None
        gc.collect()
    
    def shuffle_by_seed(self, seed):
        random.seed(seed)
        random.shuffle(self.available_indices)
    
    def _shuffle_samples(self, train_val_split=None, split_seed=42, dataset_type='train'):
        ends = np.array(self.replay_buffer.meta.episode_ends)
        sample_names = self.replay_buffer.data['sample_names'][-len(ends):]

        ends = ends - ends[0]
        starts,ends = ends[:-1], ends[1:]  
        if train_val_split is not None:
            np.random.seed(split_seed)
            indices = np.arange(len(starts))
            np.random.shuffle(indices)

            starts = starts[indices]
            ends = ends[indices]
            sample_names = sample_names[indices]

            split_idx = int(len(ends) * train_val_split)
            if dataset_type == 'train':
                ends = ends[:split_idx]
                starts = starts[:split_idx]
                sample_names = sample_names[:split_idx]
            else:
                ends = ends[split_idx:]
                starts = starts[split_idx:]
                sample_names = sample_names[split_idx:]

        self.starts = starts
        self.ends = ends
        self.sample_names = sample_names

    def __len__(self) -> int:
        if self.replay_buffer is None:
            self.load_data()
        if self.available_indices is None:
            self.available_indices = self._get_valid_indices()

        return len(self.available_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def _find_episode_name(self, frame_idx):
        raise NotImplementedError
    
    @property
    def action_start_point(self):
        '''
        动作和视频存放起始位置不同
        动作的起始位置self.replay_buffer.meta.episode_ends[0]
        视频的起始位置为0
        '''
        return self.replay_buffer.meta.episode_ends[0]
    
    def _get_valid_indices(self):
        raise NotImplementedError


class MapChunkDataset(MapDataset):
    def __init__(self,
                 lock_path:str=None,
                 cache_path:str=None,
                 zarr_path:str=None,
                 read_from_cache:bool=True,
                 flush_cache:bool=False,
                 memory_size:int=8,#GB
                 data_config:Dict=None,
                 dataset_type:str='train',
                 label_data_parser=None,
                 action_sampling_rate=1.0,
                 mask_data_parser=None,                
                 mask_type=None,
                 mask_keys=None,
                 *args, **kwargs):
        super().__init__(lock_path=lock_path,
                        cache_path=cache_path,
                        zarr_path=zarr_path,
                        read_from_cache=read_from_cache,
                        flush_cache=flush_cache,
                        memory_size=memory_size,
                        action_sampling_rate=action_sampling_rate,
                        *args, **kwargs)
            
        self.data_config = data_config
        self.dataset_name = "_".join(data_config["path"].split('/')[-2:]) if "path" in data_config else "None"
        self.dataset_type = dataset_type
        self._get_instruction = label_data_parser.get_instruction \
                                    if label_data_parser else None
        self._get_mask = None
        if mask_data_parser:
            self._get_mask = mask_data_parser.get_compressed_mask if mask_type in ['mask_as_channel', 'mask_only'] else None
            self._get_instruction = mask_data_parser.get_instruction # 用mask meta文件里的instruction替换原来的instruction
        self.mask_type = mask_type
        self.mask_keys = mask_keys
        self._get_params(data_config)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        '''
        Get a chunk from the dataset
        '''
        try:
        # if True:
            if self.replay_buffer is None:
                self.load_data()
            idx = self.available_indices[index][0]
            # print_rank0(f'idx:{idx}', flush=True)
            eposide_end = self.available_indices[index][1]
            eposide_start = self.available_indices[index][2]

            actions = {}
            action_histories = {}
            
            observations = {}
            observation_histories = {}
            observation_goal = {}
            
            action_st_idx = self.action_start_point+idx
            action_ed_idx = action_st_idx + min(self.action_horizon, eposide_end-idx)
    
            for key in self.action_keys:
                if key not in self.replay_buffer.data:
                    raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
                single_action = np.array(self.replay_buffer.data[key][action_st_idx:action_ed_idx])
                if self.right_padding:
                    pad_size = self.action_horizon - single_action.shape[0]
                    action = self._pad_with_value(single_action, pad_size, padding_dir='right')
                else:
                    action = single_action
                actions[key] = action

                action_history_start = max(action_st_idx - self.action_history_length, eposide_start+self.action_start_point)
                action_history = np.array(self.replay_buffer.data[key][action_history_start:action_st_idx])
                if self.left_padding:
                    pad_size = self.action_history_length - action_history.shape[0]
                    if action_history.shape[0] > 0:
                        action_history = self._pad_with_value(action_history, pad_size, padding_dir='left')
                    else:
                        action_history = self._pad_with_value(single_action, pad_size, padding_dir='left')
                        action_history = action_history[:self.action_history_length]
                action_histories[key] = action_history
             
            obs_st_idx, obs_ed_idx = idx, idx+self.image_horizon
            # obs_goal_idx = idx+random.choice(range(self.image_horizon, eposide_end-idx+1, 1))
            if idx+self.action_horizon<eposide_end-1:
                obs_goal_idx = idx+random.choice(range(self.action_horizon, eposide_end-idx, 1))
            else:
                obs_goal_idx = eposide_end-1
            # obs_goal_idx = eposide_end-1 # use last img as goal

            for key in self.obs_keys:
                if key not in self.replay_buffer.data:
                    raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
                obs = np.array(self.replay_buffer.data[key][obs_st_idx:obs_ed_idx])
                if self.right_padding:
                    pad_size = self.image_horizon - obs.shape[0]
                    obs = self._pad_with_value(obs, pad_size, padding_dir='right')
                observations[key] = obs

                history_start = max(obs_st_idx - self.image_history_length, eposide_start)
                # history_start = max(obs_st_idx - self.image_history_length, 0)
                obs_history = np.array(self.replay_buffer.data[key][history_start:obs_st_idx])
                if self.left_padding:
                    pad_size = self.image_history_length - obs_history.shape[0]
                    single_obs = obs[:1]
                    if obs_history.shape[0] > 0:
                        obs_history = self._pad_with_value(obs_history, pad_size, padding_dir='left')
                    else:
                        obs_history = self._pad_with_value(single_obs, pad_size, padding_dir='left')
                        obs_history = obs_history[:self.image_history_length]

                observation_histories[key] = obs_history

                obs_goal = np.array(self.replay_buffer.data[key][obs_goal_idx])
                observation_goal[key] = obs_goal
            # print(f'observations: {observations}', flush=True)
            mask = None

            frame_idx,uid = self._find_episode_name(idx)
            if self._get_instruction:
                instructions = self._get_instruction(uid=uid, frames=range(frame_idx,frame_idx+self.action_horizon))
                instruction_histories = self._get_instruction(uid=uid, frames=range(frame_idx-self.action_history_length, frame_idx))
            # print(f'{idx}: {instructions}')
            if not self._get_instruction or len(instructions['text_en']) == 0:
                instructions = {'text_en':self.default_instruction}
                instruction_histories = {'text_en':self.default_instruction}

            item_dict = {
                        'actions': actions,
                        'action_histories': action_histories,
                        'observations': observations,
                        'observation_histories': observation_histories,
                        'observation_goal': observation_goal,
                        'instructions': instructions,
                        'instruction_histories': instruction_histories,
                        'uid': uid,
                        'frame': frame_idx,
                        'dataset_name': self.dataset_name,
                }
             
            frame_idx,uid = self._find_episode_name(idx)
            if self.mask_type in ['mask_only'] and self._get_mask:
                for mask_key in self.mask_keys:
                    mask = self._get_mask(uid=uid, frame_idxs=range(frame_idx,frame_idx+self.image_horizon), mask_key=mask_key)
                    item_dict[mask_key] = mask
                    if len(mask) == 0:
                        print(f'{uid}, {mask_key} mask is []!!!', flush=True)
            
            if not self.tac_keys:
                return item_dict
            
            tactiles, tac_histories = {},{}
            for key in self.tac_keys:
                if key not in self.replay_buffer.data:
                    raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
                tac = np.array(self.replay_buffer.data[key][obs_st_idx:obs_ed_idx])
                if self.right_padding:
                    pad_size = self.image_horizon - tac.shape[0]
                    tac = self._pad_with_value(tac, pad_size, padding_dir='right')
                tactiles[key] = tac

                tac_history = np.array(self.replay_buffer.data[key][history_start:obs_st_idx])
                if self.left_padding:
                    pad_size = self.image_history_length - tac_history.shape[0]
                    tac_history = self._pad_with_value(tac_history, pad_size, padding_dir='left')
                tac_histories[key] = tac_history

            item_dict['tactiles'] = tactiles
            item_dict['tactile_histories'] = tac_histories

            return item_dict
        except:
            print(f'Error in {self.dataset_name} at index {index}', flush=True)
            return None

    def _get_params(self, data_config:Dict):
        action_horizon = data_config.get('action_horizon', 21)
        action_keys = data_config.get('action_keys', list(_ACTION_KEY_EE_MAPPING.keys()))
        right_padding = data_config.get('right_padding', True)
        left_padding = data_config.get('left_padding', True)
        action_history_length = data_config.get('action_history_length', 20)

        obs_keys = data_config.get('obs_keys', list(_CAM_FILE_MAPPING.keys()))
        image_horizon = data_config.get('image_horizon', 1)
        image_history_length = data_config.get('image_history_length', 0)

        default_instruction = data_config.get('default_instruction', None)
        train_val_split = data_config.get('train_val_split', None)
        split_seed = data_config.get('split_seed', 42)

        tac_keys = data_config.get('tac_keys', None)
        dataset_group = data_config.get('dataset_group', 'x2')

        self.action_horizon = action_horizon
        self.action_keys = action_keys
        self.right_padding = right_padding
        self.left_padding = left_padding
        self.action_history_length = action_history_length
        self.obs_keys = obs_keys
        self.image_horizon = image_horizon
        self.image_history_length = image_history_length
        self.default_instruction = default_instruction
        self.train_val_split = train_val_split
        self.split_seed = split_seed

        self.tac_keys = tac_keys
        self.parse_head_action = True
        for head_key in _HEAD_ACTION_MAPPING:
            if head_key not in self.action_keys:
                self.parse_head_action = False
        
        self.dataset_group = dataset_group

    
    def _pad_with_value(self, single_action, pad_size, padding_dir='right'):
        if pad_size > 0:
            padding_value = single_action[-1] if padding_dir == 'right' else single_action[0]
            if single_action.ndim == 1:
                padding = np.full((pad_size,), padding_value)
            else:
                padding = np.broadcast_to(padding_value, (pad_size, *single_action.shape[1:]))
            
            if padding_dir == 'right':
                padded_action = np.concatenate([single_action, padding], axis=0)
            else:
                padded_action = np.concatenate([padding, single_action], axis=0)
            return padded_action

        return single_action

        
    def _get_valid_indices(self):
        self._shuffle_samples(self.train_val_split, self.split_seed, self.dataset_type)
        if self.right_padding:
            valid_indices = [(idx, end, start) for start, end in zip(self.starts, self.ends) \
                            for idx in range(start, end, self.action_sampling_step)]

            return valid_indices

        valid_indices = [(idx, end, start) for start, end in zip(self.starts, self.ends) \
                        for idx in range(start, end, self.action_sampling_step) \
                            if idx + self.action_horizon < end]
        

        return valid_indices
    
    def _find_episode_name(self, frame_idx):
        for name, start, end in zip(*[self.sample_names, self.starts, self.ends]):
            if start <= frame_idx < end:
                return frame_idx-start,name

        return None,None

class MapFullDataset(MapDataset):
    def __init__(self,
                 lock_path:str=None,
                 cache_path:str=None,
                 zarr_path:str=None,
                 read_from_cache:bool=True,
                 flush_cache:bool=False,
                 memory_size:int=8,#GB
                 data_config:Dict=None,
                 dataset_type:str='train',
                 label_data_parser=None,
                 *args, **kwargs):
        super().__init__(lock_path=lock_path,
                        cache_path=cache_path,
                        zarr_path=zarr_path,
                        read_from_cache=read_from_cache,
                        flush_cache=flush_cache,
                        memory_size=memory_size,
                        *args, **kwargs)
        
        self.dataset_type = dataset_type
        self._get_instruction = label_data_parser.get_instruction \
                                    if label_data_parser else None

        self._get_params(data_config)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        '''
        Get a chunk from the dataset
        '''
        if self.replay_buffer is None:
            self.load_data()

        actions,observations = {},{}

        st_idx, ed_idx = self.starts[idx], self.ends[idx]
        for key in self.action_keys:
            if key not in self.replay_buffer.data:
                raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
            action = np.array(self.replay_buffer.data[key][st_idx:ed_idx])
            actions[key] = action
        
        for key in self.obs_keys:
            if key not in self.replay_buffer.data:
                raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
            obs = np.array(self.replay_buffer.data[key][st_idx:ed_idx])
            observations[key] = obs
        
        sample_name = self._find_episode_name(idx)
        
        if self._get_instruction:
            instructions = self._get_instruction(uid=sample_name, frames=[0])
        
        if not self._get_instruction or len(instructions['text_en']) == 0:
            instructions = {'text_en':self.default_instruction}

        item_dict = {
            'actions': actions,
            'observations': observations,
            'instructions': instructions,
            'uid': sample_name,
        }
        if self.tac_keys:
            tactiles = {}
            for key in self.tac_keys:
                if key not in self.replay_buffer.data:
                    raise ValueError(f'Key {key} not found in replay buffer data {self.replay_buffer}')
                tac = np.array(self.replay_buffer.data[key][st_idx:ed_idx])
                tactiles[key] = tac
            item_dict['tactiles'] = tactiles

        return item_dict

    def _get_params(self, data_config:Dict):
        default_instruction = data_config.get('default_instruction', None)
        train_val_split = data_config.get('train_val_split', None)
        split_seed = data_config.get('split_seed', 42)
        action_keys = data_config.get('action_keys', list(_ACTION_KEY_EE_MAPPING.keys()))
        obs_keys = data_config.get('obs_keys', list(_CAM_FILE_MAPPING.keys()))
        tac_keys = data_config.get('tac_keys', None)

        self.default_instruction = default_instruction
        self.train_val_split = train_val_split
        self.split_seed = split_seed

        self.action_keys = action_keys
        self.obs_keys = obs_keys
        self.tac_keys = tac_keys
        self.parse_head_action = True
        for head_key in _HEAD_ACTION_MAPPING:
            if head_key not in self.action_keys:
                self.parse_head_action = False

    def _get_valid_indices(self):
        self._shuffle_samples(self.train_val_split, self.split_seed, self.dataset_type)

        valid_indices = list(range(0, len(self.starts), self.action_sampling_step))
        return valid_indices
    
    def _find_episode_name(self, sample_idx):
        if sample_idx in self.available_indices:
            return self.sample_names[sample_idx]

        return None


def make_chunk_dataset(
        dataset_paths:List[str],
        rank=0,
        dp_world_size=1,
        dataset_buffer_size=None,
        sample_ratio:List[float]=None,
        replacement=False,
        cache_dir:str=None,
        cam_mapping=_CAM_MAPPING,
        image_height=480,
        image_width=640,
        seed=42,
        read_from_cache=False,
        memory_size=8,#GB
        flush_cache=False,
        action_sampling_rate=1.0,
        return_full_traj=False,
        data_configs:List[Dict]=None,
        read_labeled_data=True,
        parse_tactile=False,
        parse_head_action=False,
        trans_zh2en_url=None,
        num_workers=20,
        filter_angle_outliers=True,
        detect_motion=True,
        trim_stationary=False,
        mask_meta_file_path=None, # along with read_masked_data=True
        mask_type=None,
        mask_in_buffer=False, # if False, mask store in dataset, else in replaybuffer, deprecated!!!
        mask_keys=None, # used for multiple mask keys
        lang_embed_file_path=None, # used for instruction embedding
):
    
    cache_dir = cache_dir or './tmp'
    if not isinstance(cache_dir, list):
        cache_dir = [cache_dir]*len(dataset_paths)
    if not isinstance(trim_stationary,list):
        trim_stationary = [trim_stationary]*len(dataset_paths)

    mask_data_parser = MaskMetaData(file_path=mask_meta_file_path, mask_keys=mask_keys, lang_embed_file_path=lang_embed_file_path) if mask_meta_file_path else None
    for i, (dataset_path,trim,cache) in enumerate(zip(dataset_paths,trim_stationary,cache_dir)):
        if i % dp_world_size == rank:
            print(f'Rank {rank} loading {dataset_path}')
            dataset_path = os.path.normpath(dataset_path)
            last_dir = os.path.basename(dataset_path)
            second_last_dir = os.path.basename(os.path.dirname(dataset_path))
            store_id = f'{second_last_dir}_{last_dir}'

            # os.makedirs(os.path.join(cache_dir, f'{store_id}'), exist_ok=True)
            # zarr_path = os.path.join(cache_dir, f'{store_id}/0.zarr.zip')
            os.makedirs(os.path.join(cache, f'{store_id}'), exist_ok=True)
            zarr_path = os.path.join(cache, f'{store_id}/0.zarr.zip')

            if not os.path.exists(zarr_path) or flush_cache:
                print(f'{zarr_path} does not exist, creating')
                video_files = []
                for entry in glob.glob(f'{dataset_path}/*'):
                    if os.path.isdir(entry):
                        mp4_files = glob.glob(f'{entry}/*.mp4')
                        if mp4_files:
                            video_files.append(entry)

                X2ReplayBuffer.make_zarr(
                        video_files=video_files,
                        cam_mapping=cam_mapping,
                        image_height=image_height,
                        image_width=image_width,
                        compression_level=99,
                        output_dir=zarr_path,
                        parse_tactile=parse_tactile,
                        parse_head_action=parse_head_action,
                        num_workers=num_workers,
                        filter_angle_outliers=filter_angle_outliers,
                        detect_motion=detect_motion,
                        trim_stationary=trim,)
                        # trim_stationary=trim_stationary)
    
    # torch.distributed.barrier()
    
    if sample_ratio is None:
        sample_ratio = [1.0]*len(dataset_paths)
    train_datasets,val_datasets = [],[]
    train_dataset_weights,val_dataset_weights, train_data_weights,val_data_weights = [],[],[],[]

    for dataset_path, weight, data_config, cache in tqdm.tqdm(
                    zip(*[dataset_paths, sample_ratio, data_configs, cache_dir]),
                    desc='Processing dataset cache',
                    total=len(dataset_paths),
                    disable=(rank !=0)
                ):
        label_parser = None
        if read_labeled_data:
            label_file = os.path.join(dataset_path, 'labelData.json')
            if os.path.exists(label_file):
                label_parser = LabelData(label_file, default_url=trans_zh2en_url)
        

        dataset_path = os.path.normpath(dataset_path)
        last_dir = os.path.basename(dataset_path)
        second_last_dir = os.path.basename(os.path.dirname(dataset_path))
        store_id = f'{second_last_dir}_{last_dir}'


        for file in os.listdir(os.path.join(cache, f'{store_id}')):
            if not file.endswith('.zarr.zip'):
                continue
            cache_path = os.path.join(cache, f'{store_id}/0.zarr.mdb') # Not used
            lock_path = os.path.join(cache, f'{store_id}/0.lock') # Not used
            zarr_path = os.path.join(cache, f'{store_id}/{file}')
            print(f'Rank {rank} loading {zarr_path}')

            DataBuilder = MapFullDataset if return_full_traj else MapChunkDataset

            train_dataset = DataBuilder(
                    lock_path=lock_path,
                    cache_path=cache_path,
                    zarr_path=zarr_path,
                    read_from_cache=read_from_cache,
                    memory_size=memory_size,
                    flush_cache=flush_cache,
                    action_sampling_rate=action_sampling_rate,                 
                    dataset_type='train',
                    data_config=data_config,
                    label_data_parser=label_parser,
                    mask_data_parser=mask_data_parser,
                    mask_type=mask_type,
                    mask_keys=mask_keys)

            val_dataset = DataBuilder(
                    lock_path=lock_path,
                    cache_path=cache_path,
                    zarr_path=zarr_path,
                    read_from_cache=read_from_cache,
                    memory_size=memory_size,
                    flush_cache=flush_cache,
                    action_sampling_rate=action_sampling_rate,
                    dataset_type='val',
                    data_config=data_config,
                    label_data_parser=label_parser,
                    mask_data_parser=mask_data_parser if not mask_in_buffer else None,
                    mask_type=mask_type,
                    mask_keys=mask_keys)

            if len(train_dataset) > 0:
                train_data_weights += [weight]*len(train_dataset)
                train_datasets.append(train_dataset)
                train_dataset_weights.append(weight)
            
            if len(val_dataset) > 0:
                val_datasets.append(val_dataset)
                val_data_weights += [weight]*len(val_dataset)
                val_dataset_weights.append(weight)
    

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 0 else None
    
    if dataset_buffer_size is not None and dataset_buffer_size > 0:
        train_dataset = BufferedDatasetWapper(train_dataset.datasets, dataset_buffer_size, seed=seed)
        train_sampler = SlidingWindowSampler(train_dataset, dataset_weights=train_dataset_weights, replacement=replacement)
        
        val_dataset = BufferedDatasetWapper(val_dataset.datasets, dataset_buffer_size, seed=seed) if val_dataset else None
        val_sampler = SlidingWindowSampler(val_dataset, dataset_weights=val_dataset_weights, replacement=replacement) if val_dataset else None

        return (
            train_dataset,
            val_dataset,
            train_sampler,
            val_sampler
        )

    return (
        train_dataset,
        val_dataset,
        train_data_weights,
        val_data_weights
    )

class CollateHelper:
    @staticmethod
    def stack_action(actions, parse_head_action=False, fill_zero=False, to_lie_algebra=False):
        def _convert_to_lie_algebra(rotations):
            ws = []
            for euler_angles in rotations:
                w = so3_to_lie_algebra(euler_to_rotation_matrix(euler_angles))
                ws.append(w)
            return np.stack(ws, axis=0, dtype=rotations[0].dtype)
        
        convert_to_lie_algebra = _convert_to_lie_algebra if to_lie_algebra else lambda x: x

        is_bi_mode = "follow_left_ee_cartesian_pos" in actions
        right_actions = np.concatenate([
                actions["follow_right_ee_cartesian_pos"],
                convert_to_lie_algebra(actions["follow_right_ee_rotation"]),
                actions["follow_right_gripper"].reshape(-1, 1)], axis=1
            )
        
        left_actions = np.concatenate([
                actions["follow_left_ee_cartesian_pos"],
                convert_to_lie_algebra(actions["follow_left_ee_rotation"]),
                actions["follow_left_gripper"].reshape(-1, 1)], axis=1
            ) if is_bi_mode else np.zeros_like(right_actions)

        stacked_actions = np.concatenate([left_actions, right_actions], axis=1) if (is_bi_mode or fill_zero) else right_actions

        if parse_head_action and "head_actions" in actions:
            head_actions = actions['head_actions']
            stacked_actions = np.concatenate([stacked_actions, head_actions], axis=1)
        return stacked_actions
    
    @staticmethod
    def stack_obs(obs, obs_keys=None):
        if obs_keys is not None:
            observation = {key: obs[key] for key in obs_keys}
        else:
            observation = obs
        return observation

    @staticmethod
    def stack_obs_goal(obs, obs_keys=None):
        if obs_keys is not None:
            observation_goal = random.choice([obs[key] for key in obs_keys])
        else:
            observation_goal = random.choice([obs[key] for key in obs])
        
        return observation_goal

    @staticmethod
    def stack_instruction(sample, sample2instruct):
        try:
            new_instruction = sample["instructions"]['sub'][0]["text_en"]
        except:
            new_instruction = sample['instructions']["text_en"]
            
        sample_id = sample['uid']
        if sample2instruct is not None:
            if sample_id in sample2instruct:
                if isinstance(sample2instruct[sample_id], list): 
                    random_element = random.choice(sample2instruct[sample_id])
                    new_instruction = random_element
                else:
                    new_instruction = sample2instruct[sample_id]
            else:
                print(f'use global instruction: {sample_id}')
        return new_instruction
        


def full_collate_fn(batch, obs_keys, action_dim, is_bi_mode, parse_head_action=False, to_lie_algebra=False):
    batch = list(filter(lambda x: x is not None, batch))
    batched_dict = defaultdict(list)

    for sample in batch:
        actions = CollateHelper.stack_action(
                sample["actions"],
                parse_head_action=parse_head_action,
                to_lie_algebra=to_lie_algebra
            )
        batched_dict['action'].append(actions)

        observation = CollateHelper.stack_obs(sample["observations"], obs_keys)
        batched_dict['obs'].append(observation)

        batched_dict['instruction'].append(sample['instructions'])
        batched_dict['uid'].append(sample['uid'])

        batched_dict['tactile'].append(sample['tactiles']) if 'tactiles' in sample else None

        max_len = max(len(item[obs_keys[0]]) for item in batched_dict['obs'])

        def pad_sequence(seq, max_len):
            if len(seq) < max_len:
                padding_shape = (max_len - len(seq),) + (1,) * (seq.ndim - 1)
                # padding = np.expand_dims(seq[-1], axis=0)
                padding = np.tile(seq[-1], padding_shape)
                return np.concatenate((seq, padding), axis=0)
            return seq

        obs_dict = {key: torch.as_tensor(np.stack([pad_sequence(item[key], max_len) \
                            for item in batched_dict['obs']])) for key in obs_keys}
    
    actions = np.stack([pad_sequence(item, max_len) \
                        for item in batched_dict['action']])
    actions = torch.as_tensor(actions[:, :, :action_dim])

    item_dict = {'action': actions,
            'obs': obs_dict,
            "instruction": batched_dict['instruction'],
            'uid': batched_dict['uid']
        }
    if 'tactile' in batched_dict:
        tactiles = torch.concatenate([torch.as_tensor(
                            np.stack([item[key] for item in batched_dict['tactile']], axis=0)
                    ) for key in batched_dict['tactile'][0]
                ], dim=-1) # (bs, horizon, tactile_dim, tactile_num*tactile_space), 2*15

        item_dict['tactiles'] = tactiles

    return item_dict


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.3):
    """
    将mask以指定颜色和透明度叠加到RGB图像上。
    
    参数:
    - image: RGB图像的NumPy数组,形状为(T, height, width, 3)
    - mask: 与图像形状相同的mask数组,形状为(T, height, width)
    - color: 叠加的颜色,默认为(255, 0, 0),表示红色
    - alpha: 叠加的透明度,默认为0.3
    
    返回:
    - overlaid_image: 叠加后的图像数组,形状与输入图像相同
    """
    T, height, width, _ = image.shape
    mask = np.array(mask)
    # 扩展mask的维度
    # mask = np.expand_dims(mask, axis=-1)
    # mask = np.tile(mask, (1, 1, 1, 3))
    if len(mask) == 0 or len(image) == 0:
        print(f'mask:{mask}, image:{image}', flush=True)
    # print(f'image.shape:{image[0].shape}, mask.shape:{mask[0].shape}, T:{T}', flush=True)
    overlaid_images = np.zeros_like(image, dtype=np.uint8)
    for t in range(T):
        # 创建一个与图像形状相同的颜色叠加层
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay2 = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[mask[0][t] == 1] = (255,0,0)  # 红色覆盖区域
        overlay2[mask[1][t] == 1] = (0,0,255)  # 蓝色覆盖区域
        
        # 将颜色叠加层和原始图像混合
        overlaid_image = cv2.addWeighted(overlay, alpha, image[t], 1 - alpha, 0)
        overlaid_image = cv2.addWeighted(overlay2, alpha, overlaid_image, 1 - alpha, 0)
        overlaid_images[t] = overlaid_image
    return overlaid_images

def overlay_masks(mask, T):
    """
    将mask重合在一起。
    
    参数:
    - mask: mask数组,形状为(2, height, width)
    - T: 与图像相同长度的mask
    
    返回:
    - overlaid_image: 叠加后的mask数组,形状与图像相同
    """
    mask = np.array(mask)

    # 扩展mask的维度
    if len(mask) == 0:
        print(f'mask:{mask} is none', flush=True)
    # print(f'mask[0].shape:{mask[0].shape}')
    _, height, width = mask[0].shape
    # print(f'image.shape:{image[0].shape}, mask.shape:{mask[0].shape}, T:{T}', flush=True)
    overlaid_images = np.zeros((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        # 创建一个与图像形状相同的颜色叠加层
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        overlay2 = np.zeros((height, width, 3), dtype=np.uint8)
        overlay[mask[0][t] == 1] = (255,0,0)  # 红色覆盖区域
        overlay2[mask[1][t] == 1] = (0,0,255)  # 蓝色覆盖区域
        
        overlaid_image = overlay + overlay2
        overlaid_images[t] = overlaid_image
    return overlaid_images

def overlay_masks_as_channel(mask, image):
    """
    将mask作为channel和图像重合在一起。
    
    参数:
    - mask: mask数组,形状为(2, T, height, width)
    - image: (T,height, width, 3) 
    
    返回:
    - overlaid_image: 叠加后的图像
    """
    mask = np.array(mask)

    # 扩展mask的维度
    if len(mask) == 0:
        print(f'mask:{mask} is none', flush=True)
    # print(f'mask[0].shape:{mask[0].shape}')
    T, height, width = mask[0].shape
    new_mask = (mask * 255).astype(np.uint8)
    # 转置mask，使其形状为(T, height, width, 2)
    new_mask = np.transpose(new_mask, (1, 2, 3, 0))  # (T, height, width, 2)
    overlaid_images = np.concatenate([image, new_mask], axis=-1)
    return overlaid_images

def overlay_box_on_image(image, box, color=(255, 0, 0)):
    """
    将边界框可视化在视频上，并保存结果视频。
    
    参数:
    - boxes: Numpy矩阵, 形状为 (frames, 8)，表示每帧的物体边界框，8代表两个边界框的左上角 (x1, y1) 和右下角 (x2, y2)
    """
    T, height, width, _ = image.shape
    images = []
    for t in range(T):
        x_min, y_min, x_max, y_max,  xx_min, yy_min, xx_max, yy_max = box[t] 
        # print('here:', x_min, y_min, x_max, y_max,  xx_min, yy_min, xx_max, yy_max)
        # 在帧上绘制边界框
        oimage = image[t]
        oimage = cv2.rectangle(oimage, (x_min, y_min), (x_max, y_max), (255,0,0), thickness=2)  
        oimage = cv2.rectangle(oimage, (xx_min, yy_min), (xx_max, yy_max), (0,0,255), thickness=2)  
        images.append(oimage)
    return images
    
    
def chunk_collate_fn(batch,
                     obs_keys,
                     low_dim_obs_horizon,
                     img_obs_horizon,
                     horizon,
                     action_dim,
                     is_bi_mode,
                     sample2instruct,
                     to_lie_algebra=False,
                     sample2imginstruct=None,
                     parse_head_action=False,
                     mask_type=None,
                     mask_keys=None,
                     merge_cur_history=False, # 是否合并action_history到agent_pos里面去
                     merge_image_history=False, # 是否合并image_history到obs里面去
                     ):
    # mask_type: mask_only return mask, mask_on_image return image(face_view) with mask

    batch = list(filter(lambda x: x is not None, batch))
    # for grounding
    batched_img_instructions = []
    batched_static_text = []
    batched_dynamic_text = []
    # for mask
    batched_mask_dict = {mask_key:[] for mask_key in mask_keys} if mask_keys else {}
    batched_lang_embed = []
    

    batched_dict = defaultdict(list)

    for sample in batch:
        if sample2imginstruct is not None and sample['uid'] not in sample2imginstruct:
            continue

        actions = CollateHelper.stack_action(
                    sample["actions"], 
                    parse_head_action=parse_head_action,
                    fill_zero=True,
                    to_lie_algebra=to_lie_algebra
        )
        batched_dict['action'].append(actions)

        action_history = CollateHelper.stack_action(
                sample["action_histories"],
                parse_head_action=parse_head_action,
                fill_zero=True,
                to_lie_algebra=to_lie_algebra)

        batched_dict['action_history'].append(action_history)

        observation = CollateHelper.stack_obs(sample["observations"], obs_keys)
        if merge_image_history:
            observation_histories = CollateHelper.stack_obs(sample["observation_histories"], obs_keys)
            batched_dict['obs_history'].append(observation_histories)
            # print(f'obs_history:{observation_histories}, {observation_histories["face_view"].shape}')

        observation_goal = CollateHelper.stack_obs_goal(sample["observation_goal"], obs_keys)

        batched_dict['obs'].append(observation)
        batched_dict['obs_goal'].append(observation_goal)


        try:
            new_instruction = sample["instructions"]['sub'][0]["text_en"]
        except:
            new_instruction = sample['instructions']["text_en"]
            
        batched_dict['instruction'].append(new_instruction)
 
        if 'lang_embed' in sample['instructions']:
            batched_lang_embed.append(sample['instructions']['lang_embed'][0]) # [0]是因为instructions是list，长度是T

        if mask_type and mask_type != 'instruction_only':
            assert mask_type in ['mask_only', 'mask_on_image', 'box_on_image', 'mask_as_channel', 'instruction_only'], f'{mask_type} is not support!'
            for mask_key in mask_keys:
                mask = sample[mask_key]
                if mask_type == 'mask_as_channel':
                    # 不需要额外存mask
                    continue
                elif mask_type == 'mask_on_image': # not used now
                    mask = overlay_mask_on_image(image=observation['face_view'],  mask=mask)
                elif mask_type == 'box_on_image': # not used now
                    mask = overlay_box_on_image(image=observation['face_view'],  box=mask)
                elif mask_type == 'mask_only':
                    # print(f'{mask_key} mask:{mask} is none', flush=True)
                    mask = overlay_masks(mask=mask, T=img_obs_horizon)
                batched_mask_dict[mask_key].append(mask)

        batched_dict['uid'].append(sample['uid'])
        batched_dict['frame'].append(sample['frame'])
        batched_dict['tactile'].append(sample['tactiles']) if 'tactiles' in sample else None

    # print batch_size and batched_lang_embed length
    # print_rank0(f'batch_size: {len(batch)}, batched_lang_embed length: {len(batched_lang_embed)}', flush=True)

    actions = np.stack(batched_dict['action'])
    obs_dict = {key: torch.as_tensor(np.stack([item[key] for item in batched_dict['obs']])[:, :img_obs_horizon, ...]) for key in obs_keys}
    obs_dict_goal = torch.as_tensor(np.array(batched_dict['obs_goal']))
    # must before agent pos
    if merge_image_history:
        obs_history_dict = {key: torch.as_tensor(np.stack([item[key] for item in batched_dict['obs_history']])) for key in obs_keys}
        # print(f'image_history shape: {obs_history_dict["face_view"].shape}, obs shape:{obs_dict["face_view"].shape}', flush=True)
        obs_dict = {key: torch.concat([obs_history_dict[key], obs_dict[key]], dim=1) for key in obs_keys} # 把history拼在前面

    obs_dict['agent_pos'] = torch.as_tensor(actions[:, :low_dim_obs_horizon, :action_dim])
    actions = torch.as_tensor(actions[:, low_dim_obs_horizon:low_dim_obs_horizon + horizon, :action_dim])
    action_history = torch.as_tensor(np.stack(batched_dict['action_history']))
    if merge_cur_history:
        # print(f'action_history shape: {action_history.shape}, agent_pos shape:{obs_dict["agent_pos"].shape}', flush=True)
        obs_dict['agent_pos'] = torch.concat([action_history, obs_dict['agent_pos']], dim=1) 


    batched_dict = {
            'action': actions, 
            'obs': obs_dict, 
            "obs_goal": obs_dict_goal,
            "instruction": batched_dict['instruction'],
            "dataset_name": [sample['dataset_name'] for sample in batch],
            'uid': batched_dict['uid'],
            'frame': batched_dict['frame']
        }
    if not merge_cur_history:
        batched_dict['action_history'] = action_history

    if len(batched_lang_embed) > 0:
        # lang_embed shape num_token,embed_dim, padding batched_lang_embed is torch tensor, shape: batch_size, num_token, embed_dim
        # every num_token is not equal, so we need to pad it
        # print_rank0(f'len(batched_lang_embed): {len(batched_lang_embed)}', flush=True)
        max_len = max([item.shape[0] for item in batched_lang_embed])
        # print_rank0(f'batched_lang_embed[0].shape: {batched_lang_embed[0].shape}, max_len: {max_len}', flush=True)
        batched_lang_embed = [torch.cat([item, torch.zeros(max_len - item.shape[0], item.shape[1])], dim=0) for item in batched_lang_embed]
        batched_dict['lang_embed'] = torch.stack(batched_lang_embed)
        batched_dict['lang_embed_mask'] = torch.stack([torch.cat([torch.ones(item.shape[0]), torch.zeros(max_len - item.shape[0])], dim=0) for item in batched_lang_embed])
        # print(f'lang_embed shape: {batched_dict["lang_embed"].shape}, lang_embed_mask shape: {batched_dict["lang_embed_mask"].shape}', flush=True)

    if len(batched_img_instructions) > 0:
        batched_dict['img_instruction'] = torch.as_tensor(np.stack(batched_img_instructions, axis=0))
    if len(batched_dynamic_text) > 0 and len(batched_static_text) > 0:
        assert len(batched_dynamic_text) == len(batched_static_text)
        batched_dict['static_text'] = batched_static_text
        batched_dict['dynamic_text'] = batched_dynamic_text
    if mask_keys:
        for mask_key in mask_keys:
            if len(batched_mask_dict[mask_key]) > 0:
                batched_dict[mask_key] = torch.as_tensor(np.stack(batched_mask_dict[mask_key], axis=0))

    if 'tactile' in batched_dict:
        tactiles = torch.concatenate([torch.as_tensor(
                            np.stack([item[key][:low_dim_obs_horizon] for item in batched_dict['tactile']], axis=0)
                    ) for key in batched_dict['tactile'][0]
                ], dim=-1) # (bs, low_dim_obs_horizon, tactile_dim, tactile_num*tactile_space), (bs,1,3,2*15)

        batched_dict['tactiles'] = tactiles 

    return batched_dict


def chunk_collate_fn_mix(batch,
                     low_dim_obs_horizon,
                     img_obs_horizon,
                     horizon,
                     is_bi_mode=None,
                     sample2instruct=None,
                     to_lie_algebra=False,):
    batch = list(filter(lambda x: x is not None, batch))
    batched_actions, batched_obs, batched_instructions,batched_obs_goal = [], [], [], []
    batched_tactiles, batched_uid, batched_frame = [], [], []
    batched_action_history = []
    if is_bi_mode is not None:
        print("❗❗❗is_bi_mode is not used anymore")

    obs_keys = batch[0]["observations"].keys()
    for sample in batch:
        actions = CollateHelper.stack_action(sample["actions"], fill_zero=True, to_lie_algebra=to_lie_algebra)
        batched_actions.append(actions)

        action_history = CollateHelper.stack_action(sample["action_histories"], fill_zero=True)
        batched_action_history.append(action_history)       

        if obs_keys is not None:
            observation = {key: sample['observations'][key] for key in obs_keys}
            observation_goal = random.choice([sample['observation_goal'][key] for key in obs_keys])
        else:
            observation = sample['observations']
        batched_obs.append(observation)
        batched_obs_goal.append(observation_goal)

        new_instruction = CollateHelper.stack_instruction(sample, sample2instruct)
        batched_instructions.append(new_instruction)

        batched_uid.append(sample['uid'])
        batched_frame.append(sample['frame'])

        if 'tactiles' in sample:
            tactiles = sample['tactiles']
            batched_tactiles.append(tactiles)

    actions = np.stack(batched_actions)
    obs_dict = {key: torch.as_tensor(np.stack([item[key] for item in batched_obs])[:, :img_obs_horizon, ...]) for key in obs_keys}
    obs_dict_goal = torch.as_tensor(np.array(batched_obs_goal))
    obs_dict['agent_pos'] = torch.as_tensor(actions[:, :low_dim_obs_horizon, :])
    actions = torch.as_tensor(actions[:, low_dim_obs_horizon:low_dim_obs_horizon + horizon, :])
    action_history = torch.as_tensor(np.stack(batched_action_history))

    ret_items = {
        'action': actions, 
        'obs': obs_dict, 
        "obs_goal": obs_dict_goal,
        "instruction": batched_instructions,
        "action_history": action_history,
        "dataset_name": [sample['dataset_name'] for sample in batch],
        'uid': batched_uid, 
        'frame': batched_frame
    }
    
    if len(batched_tactiles) > 0:
        tactiles = torch.concatenate([torch.as_tensor(
                            np.stack([item[key][:low_dim_obs_horizon] for item in batched_tactiles], axis=0)
                    ) for key in batched_tactiles[0]
                ], dim=-1) # (bs, low_dim_obs_horizon, tactile_dim, tactile_num*tactile_space), (bs,1,3,2*15)

        ret_items['tactiles'] = tactiles
    
    return ret_items


def collate_wrapper(obs_keys = ['face_view', 'left_wrist_view', 'right_wrist_view'],
                    collate_type = 'chunking',
                    low_dim_obs_horizon=1,
                    img_obs_horizon=1,
                    horizon=20,
                    action_dim=14,
                    is_bi_mode=True,
                    sample2instruct=None,
                    to_lie_algebra=False,
                    sample2imginstruct=None,
                    parse_head_action=False,
                    mask_type=None,
                    mask_keys=None,
                    merge_cur_history=False,
                    merge_image_history=False,):
    '''
    sample2instruct is a dict, key is sample_id, value is list of instruct(str) or instruct(str)
    '''
    if collate_type == 'chunking':
        return functools.partial(chunk_collate_fn,
                          obs_keys=obs_keys,
                          low_dim_obs_horizon=low_dim_obs_horizon,
                          img_obs_horizon=img_obs_horizon,
                          horizon=horizon,
                          action_dim=action_dim,
                          is_bi_mode=is_bi_mode,
                          sample2instruct=sample2instruct,
                          to_lie_algebra=to_lie_algebra,
                          sample2imginstruct=sample2imginstruct,
                          parse_head_action=parse_head_action,
                          mask_type=mask_type,
                          mask_keys=mask_keys,
                          merge_cur_history=merge_cur_history,
                          merge_image_history=merge_image_history)
    elif collate_type == 'full':
        return functools.partial(full_collate_fn,
                          obs_keys=obs_keys,
                          action_dim=action_dim,
                          is_bi_mode=is_bi_mode,
                          parse_head_action=parse_head_action,
                          to_lie_algebra=to_lie_algebra)
    elif collate_type == 'mix':
        return functools.partial(chunk_collate_fn_mix,
                                    low_dim_obs_horizon=low_dim_obs_horizon,
                                    img_obs_horizon=img_obs_horizon,
                                    horizon=horizon,
                                    sample2instruct=sample2instruct,
                                    to_lie_algebra=to_lie_algebra,
                                    )

    return None

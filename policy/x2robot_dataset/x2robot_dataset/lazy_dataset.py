from typing import List, Tuple, Dict, Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass
import queue
import threading
import os
import random
from sympy import rem
from torch import seed
from torch.utils.data import IterableDataset,Sampler
import numpy as np
import time

import torch
import torch.distributed as dist

from collections import defaultdict
import tqdm

import multiprocessing
from dataclasses import dataclass, field

from x2robot_dataset.common.data_preprocessing import _TAC_FILE_MAPPING_V2
from x2robot_dataset.common.utils import balanced_split_between_processes
from x2robot_dataset.common.datasets import (
    create_instance,
    CLASS_REGISTRY,
    ImageDataBuilder,
    VideoDataBuilder,
    MultiVideoLazyDataset
)

import warnings
import gc

warnings.filterwarnings("ignore", message="Length of IterableDataset")


@dataclass
class X2RDataProcessingConfig:
    cam_mapping:dict            = field(default_factory=lambda: {'faceImg': 'face_view', 'leftImg': 'left_wrist_view', 'rightImg': 'right_wrist_view'})
    tac_mapping:Dict[str,str]   = field(default_factory=dict)
    # action_keys:List[str]       = field(default_factory=lambda: list(_ACTION_KEY_EE_MAPPING.keys()))
    predict_action_keys:List[str] = field(default_factory=list)
    obs_action_keys:List[str]   = field(default_factory=list)

    tac_dim:tuple               = (3,15)
    head_action_dim:tuple       = (2,)
    image_height:int            = 480
    image_width:int             = 640
    fps:int                     = 20
    filter_angle_outliers:bool  = True
    detect_motion:bool          = True
    trim_stationary:bool        = False
    parse_tactile:bool          = False
    parse_depth_image:bool      = False
    parse_head_action:bool      = False

    train_test_split:float      = 0.9
    split_seed:int              = 42

    default_instruction:str     = ''
    class_type:str              = 'x2'  # options: 'x2', 'hf', 'zarr', 'image', 'video'

    sample_rate: Optional[float] = None  # Sampling rate for both actions and images (0-1), None means no sampling
    task:str = "action_prediction"
    instruction_path:str = None
    instruction_key:List[Dict] = None
    
    low_dim_obs_horizon         = 1

    def __post_init__(self):
        if self.class_type == 'x2':
            if self.cam_mapping is None:
                self.cam_mapping = {
                    'faceImg':'face_view',
                    'leftImg':'left_wrist_view',
                    'rightImg':'right_wrist_view'
                }
            self.tac_mapping = _TAC_FILE_MAPPING_V2

        allowed_class_types = CLASS_REGISTRY.keys()
        if self.class_type not in allowed_class_types:
            raise ValueError(
                f"Invalid value for `class_type`: '{self.class_type}'. "
                f"Allowed values are: {allowed_class_types}."
            )

        # Validate sampling rate
        if self.sample_rate is not None and not 0 < self.sample_rate <= 1:
            raise ValueError(f"sample_rate must be between 0 and 1, got {self.sample_rate}")

    def as_dict(self):
        return self.__dict__

    def get(self, key: str, default=None):
        """获取配置值，如果键不存在则返回默认值"""
        return getattr(self, key, default)

    @property
    def use_6D_rotation(self):
        if hasattr(self, '_use_6D_rotation'):
            return self._use_6D_rotation
        else:
            for key in self.predict_action_keys:
                if '6D' in key:
                    self._use_6D_rotation = True
                    return True
            self._use_6D_rotation = False
            return False

    @property
    def use_relative_action(self):
        if hasattr(self, '_use_relative_action'):
            return self._use_relative_action
        else:
            for key in self.predict_action_keys:
                if 'relative' in key:
                    self._use_relative_action = True
                    return True
            self._use_relative_action = False
            return False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

@dataclass
class X2RDataChunkConfig:
    action_horizon:int          = 21
    action_history_length:int   = 0
    image_horizon:int           = 1
    image_history_length:int    = 0
    merge_cur_history:bool      = False # 是否把历史action和当前action合并
    merge_image_history:bool    = False # 是否把历史image和当前image合并
    left_padding:bool           = False
    right_padding:bool          = True
    tac_keys:List[str]          = None

    # label_data_parser and mask_data_parser should be a class that has some method
    label_data_parser:object    = None
    mask_data_parser:object     = None
    mask_type:str               = None
    mask_keys:List[str]         = None

    return_last_obs:bool        = True
    return_first_obs:bool       = False
    return_next_obs_indices:List[int]  = None # 用于返回未来observation的下标列表

    def __post_init__(self):
        if self.action_history_length > 0 and not self.left_padding:
            raise ValueError("不能同时设置 action_history_length > 0 和 left_padding = False。当需要历史动作时必须启用left_padding。")

    def as_dict(self):
        return self.__dict__

    @property
    def use_relative_action(self):
        if hasattr(self, '_use_relative_action'):
            return self._use_relative_action
        else:
            for key in self.predict_action_keys:
                if 'relative' in key:
                    self._use_relative_action = True
                    return True
            self._use_relative_action = False
            return False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

def make_lazy_dataset(*,
    data_folders: List[str],
    data_configs: List[Dict],
    rank: int = 0,
    num_threads: int = 32,
    force_overwrite: bool = False,
    save_meta_data: bool = True,
    split: str = 'train',
    root_dir = None,
    accelerator = None,
):
    assert len(data_folders) == len(data_configs), "Mismatched data folders and configs"
    
    if accelerator is not None:
        rank = accelerator.process_index
        num_processes = accelerator.num_processes

    data_pairs = list(zip(data_folders, data_configs))
    
    repos, splits = [], []
    
    # torch.cuda.cudart().cudaProfilerStart()
    # print(f"rank: {rank} | loading dataset...", flush=True)
    # print(f"rank: {rank} | len_data: {len(data_pairs)}, num_processes: {num_processes}", flush=True)
    if accelerator is not None:
        with balanced_split_between_processes(
            inputs=data_pairs,
            num_processes=num_processes,
            process_index=rank,
            apply_padding=False  # 禁用自动填充
        ) as local_pairs:
            for idx, (dataset_path, single_config) in enumerate(local_pairs):
                print(f"[rank {rank} - ({idx+1}/{len(local_pairs)})] Processing {dataset_path}", flush=True)
                start_time = time.time()
                out_repo = create_instance(single_config).from_raw_to_videolazy_format(
                    dataset_path=dataset_path,
                    force_overwrite=force_overwrite,
                    save_meta_data=save_meta_data,
                    num_threads=num_threads,
                    root_dir=root_dir,
                    class_type=single_config['class_type']
                )
                out_repo['config'] = single_config
                
                percentage = single_config['train_test_split'] * 100
                split_expr = f'train[:{percentage}%]' if split == 'train' else f'train[{percentage}%:]'
                
                repos.append(out_repo)
                splits.append(split_expr)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"[rank {rank} - ({idx+1}/{len(local_pairs)})] Finished processing {dataset_path} in {elapsed_time:.2f} seconds", flush=True)
    else:
        local_pairs = data_pairs
        for idx, (dataset_path, single_config) in enumerate(local_pairs):
            print(f"[rank {rank} - ({idx+1}/{len(local_pairs)})] Processing {dataset_path}", flush=True)
            
            out_repo = create_instance(single_config).from_raw_to_videolazy_format(
                dataset_path=dataset_path,
                force_overwrite=force_overwrite,
                save_meta_data=save_meta_data,
                num_threads=num_threads,
                root_dir=root_dir,
                class_type=single_config['class_type']
            )
            out_repo['config'] = single_config
            
            percentage = single_config['train_test_split'] * 100
            split_expr = f'train[:{percentage}%]' if split == 'train' else f'train[{percentage}%:]'
            
            repos.append(out_repo)
            splits.append(split_expr)

    # accelerator.wait_for_everyone()
    # print(f"rank: {rank} | dataset loaded", flush=True)
    # torch.cuda.cudart().cudaProfilerStop()
    if accelerator is not None:
        from torch.distributed import all_gather_object

        all_repos = [None] * accelerator.num_processes
        all_splits = [None] * accelerator.num_processes
        all_gather_object(all_repos, repos)
        all_gather_object(all_splits, splits)

        repos = [item for sublist in all_repos if sublist is not None for item in sublist]
        splits = [item for sublist in all_splits if sublist is not None for item in sublist]
    else:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    # torch.cuda.cudart().cudaProfilerStart()
    dataset = MultiVideoLazyDataset(repos=repos, root=root_dir, split=splits)
    # accelerator.wait_for_everyone()
    # torch.cuda.cudart().cudaProfilerStop()
    return dataset


class IterChunkDataset(IterableDataset):
    """Dataset for loading and iterating over video frames with preloading support."""

    def __init__(self, 
            data_folders: List[str], 
            data_configs: List[X2RDataProcessingConfig],
            data_chunk_config:X2RDataChunkConfig,
            preload_pool_size: int = 128,
            num_preloader_threads: int = 64,
            max_frame_buffer_size: int = 128,
            num_frame_producer_threads = 5,
            accelerator = None,
            rank: int = 0,
            force_overwrite: bool = False,
            save_meta_data: bool = True,
            root_dir:str = Path('/x2robot/Data/.cache/hf_datasets'),
            split:str = 'train',
            world_size:int = 1,
            slice_size:int = 1, # batch size per rank
            insert_func:Callable = None, # do anything that is not included in default __iter__
            epoch:int = 0,
            action_keys: List[str] = None,
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            file_paths: List of paths to video files
        """
        self.rank = rank
        self.world_size = world_size
        self.data_folders = data_folders
        self.data_configs = data_configs
        self.data_chunk_config = data_chunk_config
        self.insert_func = insert_func
        self.accelerator = accelerator
        self.action_keys = action_keys

        self._get_instruction = None
        self._get_mask = None
        self.mask_type, self.mask_keys = None, None
        self.update_instruction_and_mask_parsers()

        # 加载dataset
        self.dataset = make_lazy_dataset(
            data_folders=data_folders,
            data_configs=data_configs,
            rank=rank,
            accelerator=accelerator,
            num_threads=num_preloader_threads,
            force_overwrite=force_overwrite,
            save_meta_data=save_meta_data,
            root_dir=root_dir,
            split=split
        )

        self.sampler = self.BalancedSampler(
                    frame_counts=self.dataset.frame_count_list,
                    world_size=world_size,
                    rank=rank,
                    epoch=epoch,
                    shuffle=True,
                )
        # 每个rank对总体batch贡献的范围
        self.true_bs = world_size * slice_size
        self.process_slice = range(self.rank * slice_size, (self.rank + 1) * slice_size)
        self.rank_accum_frame_count = 0

        # frame buffer pool
        self.frame_buffer = []
        self.frame_buffer_lock = threading.Lock()
        self.frame_buffer_condition = threading.Condition(self.frame_buffer_lock)
        self.max_frame_buffer_size = max_frame_buffer_size
        self.num_frame_producer_threads = num_frame_producer_threads
        
        # preload pool
        self.preload_pool_size = preload_pool_size
        self.num_preloader_threads = num_preloader_threads
        assert self.preload_pool_size >= self.num_frame_producer_threads, \
                f"preload_pool_size should be larger or equal to num_frame_producer_threads, \
                now: preload_pool_size - {self.preload_pool_size} | \
                num_frame_producer_threads - {self.num_frame_producer_threads}"
        self.preload_queue = queue.Queue(maxsize=self.preload_pool_size)
        self.file_list_queue = queue.Queue()
        self.none_count = multiprocessing.Value('i', 0)

        self.should_stop = threading.Event()
        self._initialize_data_loading()

    def __len__(self) -> int:
        return self.dataset.num_episodes

    @property
    def num_frames(self):
        # return self.cliped_frame_count
        return self.sampler.rank_min_frame_count*self.world_size 

    def _initialize_data_loading(self) -> None:
        """Initialize data loading by starting preloader threads and queueing files."""
        """Start the preloader threads."""
        self.preloader_threads = []
        for i in range(self.num_preloader_threads):
            thread = threading.Thread(target=self._preloader_worker, daemon=True, name=f"preloader_{i}")
            thread.start()
            self.preloader_threads.append(thread)
        # Queue files for processing and add termination signals.
        for indices in self.sampler:
            self.file_list_queue.put(indices)
                
        # Add termination signals
        for _ in range(self.num_preloader_threads):
            self.file_list_queue.put(None)

    def _preloader_worker(self) -> None:
        """Worker function for preloading and processing videos."""
        while not self.should_stop.is_set():  # 检查停止标志
            try:
                idx = self.file_list_queue.get(timeout=0.1)
                if idx is None:
                    self.preload_queue.put(None)
                    print(f"rank: {self.rank} | exiting preloader, self.preload_queue:", self.preload_queue.qsize(),flush=True)
                    break
                sample = self.dataset[idx]
                self.preload_queue.put(sample)
                sample = None
                # gc.collect()
            except queue.Empty:
                continue
    
    def __iter__(self) -> Any:
        """Iterate over the dataset.
        return Dict[str, Any]: A dictionary, containing keys
        ['frame',
        'actions',
        'observations',
        'observation_goal',
        'instruction',
        'instruction_histories',
        'observation_histories',
        'action_histories',
        'tactiles',
        'tactile_histories',
        'mask',  # from _generate_frames
        ]
        
        sample from preloaded_queue:   
        ['dataset_index',
        'default_instruction',
        'dataset_name',
        'trim_stationary',  # from lazy_dataset.MultiVideoLazyDataset
        'actions.*',
        'observations.*',
        'tactiles.*',
        'trim_start',
        'trim_end',
        'length',
        'sample_name' # from hf_dataset.LazyDataBuilder]
        ]
        """
        self._start_producers()

        while True:
            frame = self._get_frame_from_buffer()
            if frame is None:
                break
            # print(f"[rank {self.rank}] yield frame: {frame}")
            yield frame
    
    def _start_producers(self):
        """启动生产者线程"""
        self.producers = []
        for i in range(self.num_frame_producer_threads):
            thread = threading.Thread(target=self._frame_producer, daemon=True, name=f"frame_producer_{i}")
            thread.start()
            self.producers.append(thread)
    
    def _get_frame_from_buffer(self):
        with self.frame_buffer_condition:
            while not self.frame_buffer and self.none_count.value < self.num_preloader_threads:
                # print("frame_buffer is empty. Waiting for producers...", flush=True)
                self.frame_buffer_condition.wait()  # 等待生产者填充数据

            if not self.frame_buffer and self.none_count.value >= self.num_preloader_threads:
                # print("No more frames to process. Exiting consumer.", flush=True)
                return None  # 如果生产者完成且 frame_buffer 为空，退出循环
            
            frame_idx = random.randint(0, len(self.frame_buffer) - 1)
            frame = self.frame_buffer.pop(frame_idx)
            self.frame_buffer_condition.notify_all()  # 通知生产者有空间可用
            return frame

    def _frame_producer(self):
        """多线程生产者，负责向 frame_buffer 塞数据"""
        try:
            while not self.should_stop.is_set():  # 检查停止标志
                try:
                    sample = self.preload_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.1)
                    continue

                if self.should_stop.is_set():
                    break

                if sample is None:
                    if self._notify_none():
                        break
                    continue
                with torch.cuda.nvtx.range("generate_frames"):
                    frames = self._generate_frames(sample)
                if not frames:
                    continue

                # 安全地往 frame_buffer 中添加数据，并限制长度
                if not self._add_frames_to_buffer(frames):
                    break
        finally:
            # 确保线程结束时清理资源
            print(f"[Rank {self.rank}] Exiting producer thread", flush=True)
            # gc.collect()
    
    def _notify_none(self) -> bool:
        with self.none_count.get_lock():  # 加锁保护计数器
            self.none_count.value += 1
            # print(f"Producer received None. none_count = {self.none_count.value}", flush=True)
            if self.none_count.value >= self.num_frame_producer_threads:
                # 如果所有线程完成任务，通知所有等待的消费者线程退出
                if not self.should_stop.is_set():
                    with self.frame_buffer_condition:
                        # print(f"All producers have finished. Notifying consumers. frame_buffer {len(self.frame_buffer)} rank {self.rank}", flush=True)
                        self.frame_buffer_condition.notify_all()
                return True
        return False

    def _add_frames_to_buffer(self, frames: List[Dict]) -> bool:
        with self.frame_buffer_condition:
            while len(self.frame_buffer) >= self.max_frame_buffer_size:
                if self.should_stop.is_set():
                    break
                self.frame_buffer_condition.wait(timeout=1.0)  # 等待消费者取走数据
            
            if self.should_stop.is_set():
                return False

            self.frame_buffer.extend(frames)  # 添加新数据

            if not self.should_stop.is_set():
                self.frame_buffer_condition.notify_all()  # 通知消费者有新数据可用
        return True

    def _generate_frames(self, sample: Optional[Dict]) -> list[dict]:

        def convert_numpy_to_torch(obj):
            """递归将numpy数组转换为torch张量"""
            if isinstance(obj, dict):
                return {k: convert_numpy_to_torch(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_torch(elem) for elem in obj]
            elif isinstance(obj, np.ndarray):
                return torch.from_numpy(obj)
            else:
                return obj

        if sample is None:
            return []
        
        ######### support image only dataset ###########
        # don't need to chunk the data
        if sample['register_type'] in [ImageDataBuilder.REGISTER_KEY, VideoDataBuilder.REGISTER_KEY]:
            return [sample]
        
        indices = self._generate_frame_indices(sample) # indices of frames belonging to one sample
        length = sample['length']

        # 如果剩余的帧数不足以填充一个数据块，则返回remaining_frames个frame
        remaining_frames = self.sampler.rank_min_frame_count - self.rank_accum_frame_count
        indices = indices[:remaining_frames] if remaining_frames > 0 else []
        self.rank_accum_frame_count += len(indices)

        frames = []
        # out = True
        for frame_idx in indices:
            frame = {}

            action_st = frame_idx
            action_ed = frame_idx + min(self.data_chunk_config.action_horizon, length-frame_idx)

            obs_st = frame_idx
            obs_ed = frame_idx + min(self.data_chunk_config.image_horizon, length-frame_idx)

            if any(key.startswith('actions') for key in sample.keys()):
                frame['actions'] = {}
                frame['action_histories'] = {}

            if any(key.startswith('observations') for key in sample.keys()):
                frame['observations'] = {}
                frame['observation_histories'] = {}
                if self.data_chunk_config.return_last_obs:
                    frame['observation_goal'] = {}
                if self.data_chunk_config.return_first_obs:
                    frame['observation_first'] = {}
                if self.data_chunk_config.return_next_obs_indices:
                    frame['next_observation'] = {}

            if any(key.startswith('tactiles') for key in sample.keys()):
                frame['tactiles'] = {}
                frame['tactile_histories'] = {}

            if any(key.startswith('actions') and key.endswith('_cur') for key in sample.keys()):
                frame['joint_cur'] = {}
                frame['joint_cur_histories'] = {}

            for key in sample.keys():
                if sample[key] is None:
                    continue

                key_suffix = key.split('.')[-1]
                if key.startswith('actions') and key_suffix in self.action_keys:
                    action = np.array(sample[key][action_st:action_ed])

                    if self.data_chunk_config.right_padding:
                        pad_size = self.data_chunk_config.action_horizon - action.shape[0]
                        action = self._pad_with_value(action, pad_size, padding_dir='right')
                    frame['actions'][key_suffix] = action

                    action_history_st = max(action_st - self.data_chunk_config.action_history_length, 0)
                    action_history = np.array(sample[key][action_history_st:action_st])
                    if self.data_chunk_config.left_padding and self.data_chunk_config.action_history_length > 0:
                        pad_size = self.data_chunk_config.action_history_length - action_history.shape[0]

                        if action_history.shape[0] > 0:
                            action_history = self._pad_with_value(action_history, pad_size, padding_dir='left')
                        else:
                            action_history = self._pad_with_value(action, pad_size, padding_dir='left')
                            action_history = action_history[:self.data_chunk_config.action_history_length]

                    frame['action_histories'][key_suffix] = action_history
                
                if key.startswith('observations'):
                    obs = sample[key][obs_st:obs_ed]
                    if self.data_chunk_config.right_padding:
                        pad_size = self.data_chunk_config.image_horizon - obs.shape[0]
                        obs = self._pad_with_value(obs, pad_size, padding_dir='right')
                    frame['observations'][key_suffix] = obs

                    obs_history_st = max(obs_st - self.data_chunk_config.image_history_length, 0)
                    obs_history = sample[key][obs_history_st:obs_st]
                    if self.data_chunk_config.left_padding and self.data_chunk_config.image_history_length > 0:
                        pad_size = self.data_chunk_config.image_history_length - obs_history.shape[0]

                        if obs_history.shape[0] > 0:
                            obs_history = self._pad_with_value(obs_history, pad_size, padding_dir='left')
                        else:
                            obs_history = self._pad_with_value(obs, pad_size, padding_dir='left')
                            obs_history = obs_history[:self.data_chunk_config.image_history_length]
                
                    if obs_history.shape[0] == 0:
                        if 'observation_histories' in frame.keys():
                            frame.pop('observation_histories')
                    else:
                        frame['observation_histories'][key_suffix] = obs_history

                    if self.data_chunk_config.return_last_obs:
                        obs_goal_idx = length-1
                        frame['observation_goal'][key_suffix] = sample[key][obs_goal_idx]
                    
                    if self.data_chunk_config.return_first_obs:
                        obs_idx = indices[0]
                        frame['observation_first'][key_suffix] = sample[key][obs_idx]
                    
                    if self.data_chunk_config.return_next_obs_indices: # 用于返回未来的observation
                        frame['next_observation'][key_suffix] = []
                        for next_obs_idx in self.data_chunk_config.return_next_obs_indices:
                            next_obs_frame_idx = frame_idx + next_obs_idx
                            if next_obs_frame_idx >= length: # 相当于right_padding
                                next_obs_frame_idx = length-1
                            frame['next_observation'][key_suffix].append(sample[key][next_obs_frame_idx])
                
                if key.startswith('tactiles'):
                    key_suffix = key.split('.')[-1]

                    obs_history_st = max(obs_st - self.data_chunk_config.image_history_length, 0)
                    tac = np.array(sample[key][obs_st:obs_ed])
                    if self.data_chunk_config.right_padding:
                        pad_size = self.data_chunk_config.image_horizon - tac.shape[0]
                        tac = self._pad_with_value(tac, pad_size, padding_dir='right')
                    frame['tactiles'][key_suffix] = tac # key_suffix = 'left_tactile' or 'right_tactile'

                    tac_history = sample[key][obs_history_st:obs_st]
                    if self.data_chunk_config.left_padding and self.data_chunk_config.action_history_length > 0:   
                        pad_size = self.data_chunk_config.image_history_length - tac_history.shape[0]
                        tac_history = self._pad_with_value(tac_history, pad_size, padding_dir='left')
                    frame['tactile_histories'][key_suffix] = tac_history

                if key.startswith('actions') and key.endswith("_cur"):
                    # 关节电流/力矩
                    key_suffix = key.split('.')[-1]
                    obs_history_st = max(obs_st - self.data_chunk_config.image_history_length, 0)
                    curs = np.array(sample[key][obs_st:obs_ed])
                    if self.data_chunk_config.right_padding:
                        pad_size = self.data_chunk_config.image_horizon - curs.shape[0]
                        curs = self._pad_with_value(curs, pad_size, padding_dir='right')
                    frame['joint_cur'][key_suffix] = curs
                    curs_history = sample[key][obs_history_st:obs_st]
                    if self.data_chunk_config.left_padding and self.data_chunk_config.action_history_length > 0:
                        pad_size = self.data_chunk_config.image_history_length - curs_history.shape[0]
                        curs_history = self._pad_with_value(curs_history, pad_size, padding_dir='left')
                    frame['joint_cur_histories'][key_suffix] = curs_history
                
                if key.startswith('conditions.image'): # conditions.image.left_wrist_view
                    frame_indices = sample[key]['frame_indices']
                    if frame_idx in frame_indices:
                        key_suffix = key.split('.')[-1]
                        cond_img = np.array(sample[key]['fig'][frame_indices.index(frame_idx)])
                        frame['conditions.image'][key_suffix] = cond_img
                
                if key.startswith('conditions.video'):
                    key_suffix = key.split('.')[-1]
                    cond_vid = np.array(sample[key][frame_idx])
                    frame['conditions.video'][key_suffix] = cond_vid
                
                if key.startswith('conditions.text'):
                    key_suffix = key.split('.')[-1]
                    annotations = sample[key]['annotations']
                    frame_indices = sample[key]['frame_indices']
                    if frame_idx in frame_indices:
                        cond_text = annotations[frame_indices.index(frame_idx)]                    
                        frame['conditions.text'][key_suffix] = cond_text
            
            mask = None
            replaybuffer_mask_keys = ['mask']
            if self.mask_type == 'mask_only' and not self._get_mask: # 如果是直接从replaybuffer里取数据
                for key in replaybuffer_mask_keys:
                    if key not in self.sample:
                        raise ValueError(f'Key {key} not found in hf_dataset {sample.keys()}')
                    mask = sample[key][obs_st:obs_ed]
                    if self.right_padding:
                        pad_size = self.data_chunk_config.image_horizon - mask.shape[0]
                        mask = self._pad_with_value(mask, pad_size, padding_dir='right')

            uid = sample['sample_name']
            frame['uid'] = uid
            
            ## get instruction: 
            if "instruction" in sample:
                instruction_info = self.get_instruction_info(sample, frame_idx)
                for key, value in instruction_info.items():
                    if key == "instructions" or key == "instruction_histories":
                        frame[key] = {"text_en":value}
                    else:
                        frame[key] = value
                    
            else:
                if self._get_instruction:
                    instructions = self._get_instruction(
                        uid=uid,
                        frames=range(frame_idx,frame_idx+self.data_chunk_config.action_horizon)
                    )
                    instruction_histories = self._get_instruction(
                        uid=uid,
                        frames=range(frame_idx-self.data_chunk_config.action_history_length, frame_idx)
                    )
                
                if not self._get_instruction or len(instructions['text_en']) == 0:
                    instructions = {'text_en':sample['default_instruction']}
                    instruction_histories = instructions

                frame['instructions'] = instructions
                frame['instruction_histories'] = instruction_histories
            
            frame['frame'] = frame_idx
            frame['dataset_name'] = sample['dataset_name']

            if self.mask_type and self._get_mask:
                for mask_key in self.mask_keys:
                    mask = self._get_mask(
                        uid = uid,
                        frame_idxs = range(frame_idx, frame_idx+self.data_chunk_config.image_horizon),
                        mask_key = mask_key
                    )
                    frame[mask_key] = mask
                    if len(mask) == 0:
                        print(f'{uid}, {mask_key} mask is []!!!', flush=True)
            elif self.mask_type == 'mask_only': # used for mask stored in replaybuffer, not used now
                frame['mask'] = np.moveaxis(mask,0,1) # [2,T,H,W]

            # # if self.depth_image:
            # frame['left_depth_image'] = sample['left_depth_video'][frame_idx]
            # frame['right_depth_image'] = sample['right_depth_video'][frame_idx]

            if self.insert_func:
                ## do anything that is not included above
                frame = self.insert_func(frame)
            frame['register_type'] = sample['register_type']
            frame = convert_numpy_to_torch(frame)
            frames.append(frame)
            # if out and self.rank == 0:
            #     print(f"frame:\n {frame}", flush=True)
            #     out = False
        return frames

    def get_instruction_info(self, sample, frame_idx):
        instruction_info = {}
        if "instruction" in sample:
            instruction_info["instructions"] = sample['instruction']
        if "detailed_instruction" in sample:
            instruction_info["detailed_instruction"] = sample['detailed_instruction']

        instruction_history = []
        if "subtask_generation" in sample and sample['subtask_generation'] is not None:
            subtask_generation = {k:v for k,v in sample['subtask_generation'].items() if v is not None}
            for split in subtask_generation:
                start_index, end_index = split.split(' ')
                if frame_idx > int(end_index):
                    instruction_history.append(subtask_generation[split])
                if int(start_index) <= frame_idx < int(end_index):
                    instruction_info["subtask_generation"] = subtask_generation[split]
                    break
            instruction_info["instruction_histories"] = "\n".join(instruction_history)
        
        info_keys = ["motion_language", "distribute", "history_input", "history_output", "reasoning"]
        for k in info_keys:
            if k in sample and sample[k] is not None:
                # motion_language = {k:v for k,v in sample[k].items() if v is not None}
                info = {k:v for k,v in sample[k].items() if v is not None}
                for split in info:
                    start_index, end_index = split.split(' ')
                    if int(start_index) <= frame_idx < int(end_index):
                        instruction_info[k] = info[split]
                        break

        return instruction_info

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

    def _generate_frame_indices(self, sample: Dict) -> List[int]:
        """Generate frame indices for iteration."""
        assert 'trim_start' in sample, "Trim start not found in samples"
        assert 'trim_end' in sample, "Trim end not found in samples"
        trim_start = sample['trim_start'] if sample['trim_stationary'] else 0
        trim_end = sample['trim_end'] if sample['trim_stationary'] else sample['length']

        indices = [ frame_idx for frame_idx in range(trim_start, trim_end)]

        if not self.data_chunk_config.right_padding:
            indices = [ frame_idx for frame_idx in indices
                        if frame_idx + self.data_chunk_config.action_horizon < trim_end]
        
        if not self.data_chunk_config.left_padding:
            indices = [ frame_idx for frame_idx in indices
                        if frame_idx - self.data_chunk_config.action_history_length >= trim_start]
            

        return indices

    def update_instruction_and_mask_parsers(self):
        if self.data_chunk_config.label_data_parser:
            self._get_instruction = self.data_chunk_config.label_data_parser.get_instruction 

        if self.data_chunk_config.mask_data_parser:
            if self.data_chunk_config.mask_type in ['mask_on_image', 'mask_only']:
                self._get_mask = self.data_chunk_config.mask_data_parser.get_compressed_mask
            else:
                self._get_mask = self.data_chunk_config.mask_data_parser.get_box
            self._get_instruction = self.data_chunk_config.mask_data_parser.get_instruction
        self.mask_type = self.data_chunk_config.mask_type
        self.mask_keys = self.data_chunk_config.mask_keys

    def reset_epoch(self, epoch):
        self.clear_threads()

        # 重置停止标志
        self.should_stop.clear()
        
        # 重新初始化采样器和计数器
        # self.sampler = RandomSampler(self.ranked_index, epoch=epoch, shuffle=True)
        self.sampler = self.BalancedSampler(
                    frame_counts=self.dataset.frame_count_list,
                    world_size=self.world_size,
                    rank=self.rank,
                    epoch=epoch,
                    shuffle=True,
                )
        self.rank_accum_frame_count = 0
        
        # frame buffer pool
        self.frame_buffer = []
        self.frame_buffer_lock = threading.Lock()
        self.frame_buffer_condition = threading.Condition(self.frame_buffer_lock)

        # preload pool
        self.preload_queue = queue.Queue(maxsize=self.preload_pool_size)
        self.file_list_queue = queue.Queue()
        self.none_count = multiprocessing.Value('i', 0)
        # 重新初始化数据加载
        self._initialize_data_loading()

    def clear_threads(self):
        # 设置停止标志
        self.should_stop.set()
        
        # 清空队列并发送终止信号
        self._clear_queues()
        # 等待线程结束
        if hasattr(self, 'preloader_threads'):
            for preloader in self.preloader_threads:
                preloader.join(timeout=5.0)
                if preloader.is_alive():
                    print(f"❌ Unable to stop preloader {preloader.name} rank: {self.rank}", flush=True)

        if hasattr(self, 'producers'):
            for producer in self.producers:
                producer.join(timeout=5.0)
                if producer.is_alive():
                    print(f"❌ Unable to stop producer {producer.name} rank: {self.rank}", flush=True)

    def _clear_queues(self):
        """清空所有队列并发送终止信号"""
        # 清空 file_list_queue
        while not self.file_list_queue.empty():
            try:
                self.file_list_queue.get_nowait()
            except queue.Empty:
                break
                
        # 清空 preload_queue
        while not self.preload_queue.empty():
            try:
                self.preload_queue.get_nowait()
            except queue.Empty:
                break
                
        # 发送终止信号给 preloader 线程
        for _ in range(self.num_preloader_threads):
            self.file_list_queue.put(None)
            
        # 发送终止信号给 producer 线程
        for _ in range(self.num_frame_producer_threads):
            self.preload_queue.put(None)

    def __del__(self):
        """添加析构函数来清理资源"""
        # 清理队列
        if hasattr(self, 'preload_queue'):
            while not self.preload_queue.empty():
                try:
                    self.preload_queue.get_nowait()
                except queue.Empty:
                    break
        if hasattr(self, 'file_list_queue'):
            while not self.file_list_queue.empty():
                try:
                    self.file_list_queue.get_nowait()
                except queue.Empty:
                    break
                
        # 清理缓冲区
        if hasattr(self, 'frame_buffer'):
            self.frame_buffer.clear()
        
        # 重置计数器
        if hasattr(self, 'none_count'):
            with self.none_count.get_lock():
                self.none_count.value = 0
            
        # 清理数据集引用
        if hasattr(self, 'dataset'):
            del self.dataset

    class BalancedSampler(Sampler):
        def __init__(self,
                    frame_counts: list[list[int]], # 2d list, frame counts for each episode in each dataset
                    world_size: int,
                    rank: int,
                    epoch: int = 0,
                    shuffle: bool = True,
                    max_episode: int = None):
            # flatten 2d list to 1d list
            frame_counts = [item for sublist in frame_counts for item in sublist] # 1维列表，存储每个episode的frame数量
            frame_counts = frame_counts[:max_episode] if max_episode else frame_counts
            indices = list(range(len(frame_counts)))
            
            # 按帧数降序排列
            self.sorted_indices = sorted(indices, 
                key=lambda x: -frame_counts[x])
            
            # 贪心分配
            allocations = [[] for _ in range(world_size)]
            counts = [0]*world_size
            for idx in self.sorted_indices:
                target = counts.index(min(counts))
                allocations[target].append(idx)
                counts[target] += frame_counts[idx]
            
            self.indices = allocations[rank]
            self.balanced_frame_counts = counts

            assert self.rank_min_frame_count > 0, "At least one episode should be assigned to each rank"

            self.epoch = epoch
            self.shuffle = shuffle
            
        def __iter__(self):
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                random.seed(self.epoch)
                random.shuffle(self.indices)

            return iter(self.indices)
        
        def __len__(self):
            return len(self.indices)
        
        def set_epoch(self, epoch):
            self.epoch = epoch

        @property
        def rank_min_frame_count(self):
            return min(self.balanced_frame_counts)
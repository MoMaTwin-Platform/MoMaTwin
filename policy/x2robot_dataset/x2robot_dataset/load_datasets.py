
import torch
from filelock import FileLock
import shutil
import pathlib
from typing import Dict
import random

from x2robot_dataset.common.replay_buffer import ReplayBuffer
from x2robot_dataset.common.data_utils import (
    split_dataset_with_list,
    print_rank0
)
from x2robot_dataset.common.data_preprocessing import (
    LabelData,
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING
)
import zarr
import numpy as np
import os

from x2robot_dataset.replay_buffer import X2ReplayBuffer
from x2robot_dataset.map_dataset import MapChunkDataset, MapFullDataset

from typing import List, Dict
from torch.utils.data import Sampler, ConcatDataset ,BatchSampler
import tqdm
import gc
import psutil
import inspect
import glob

class DataGroupBatchSampler(BatchSampler):
    """
    Sampler that retrieves data batches from a dataset group
    """

    def __init__(self, dataset, batch_size):
        assert isinstance(dataset, ConcatDataset), \
            "dataset should be of type ConcatDataset"
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_indices = self.get_dataset_indices()
        self.sampler_id = "DataGroupBatchSampler"
      
    def get_dataset_indices(self):
        """
        Get indices for each dataset in ConcatDataset
        """
        # indices = []
        start_idx = 0
        grouped_indices = {}
        for dataset in self.dataset.datasets:
            dataset_group = dataset.dataset_group
            if dataset_group not in grouped_indices:
                grouped_indices[dataset_group] = []
            end_idx = start_idx + len(dataset)
            indx = list(range(start_idx, end_idx))
            random.shuffle(indx)
            grouped_indices[dataset_group].extend(indx)
            start_idx = end_idx
        for key in grouped_indices:
            print(f"dataset_group: {key}, count: {len(grouped_indices[key])}")
        all_indices = []
        for key in grouped_indices:
            dataset_indices = grouped_indices[key]
            for i in range(0, len(dataset_indices), self.batch_size):
                if i + self.batch_size < len(dataset_indices):
                    all_indices.append(dataset_indices[i:i + self.batch_size])
        print(f"all_indices count: {len(all_indices)}")
        random.shuffle(all_indices)

        return all_indices
    
    def __iter__(self):
        for indices in self.dataset_indices:
            yield indices

    def __len__(self):
        return len(self.dataset_indices)
    
def make_chunk_dataset_from_config(
    data_configs:List[Dict],
    dataload_args:Dict=None,
    ):

    ### 默认已经有cache file，不需要生成 ###
    ### 没有cache file，先调用make_chunk_dataset生成cache file ###
    
    trans_zh2en_url = dataload_args.get("trans_zh2en_url",None)
    return_full_traj = dataload_args.get("return_full_traj",False)

    torch.distributed.barrier()

    train_datasets,val_datasets = [],[]
    train_dataset_weights,val_dataset_weights, train_data_weights,val_data_weights = [],[],[],[]

    for config in tqdm.tqdm(data_configs,desc='Loading datasets',):

        dataset_path = config['path']
        cache = config['cache_file']

        weight = config.get('sample_ratio',1.0)
        read_labeled_data = config.get('read_labeled_data',True)
        action_sampling_rate = config.get('action_sampling_rate',1.0)

        label_parser = None
        if read_labeled_data:
            label_file = os.path.join(dataset_path, 'labelData.json')
            if os.path.exists(label_file):
                label_parser = LabelData(label_file, default_url=trans_zh2en_url)
        
        dataset_path = os.path.normpath(dataset_path)

        cache_path = None # Not used
        lock_path = None # Not used
        zarr_path = cache
        if not zarr_path.endswith('.zarr.zip') or not os.path.exists(zarr_path):
            raise ValueError(f'Invalid cache path {zarr_path}')
        

        DataBuilder = MapFullDataset if return_full_traj else MapChunkDataset

        read_from_cache = False # Not used
        memory_size = 0 # Not used
        flush_cache = False # Not used

        train_dataset = DataBuilder(
                lock_path=lock_path,
                cache_path=cache_path,
                zarr_path=zarr_path,
                read_from_cache=read_from_cache,
                memory_size=memory_size,
                flush_cache=flush_cache,
                action_sampling_rate=action_sampling_rate,                 
                dataset_type='train',
                data_config=config,
                label_data_parser=label_parser)

        val_dataset = DataBuilder(
                lock_path=lock_path,
                cache_path=cache_path,
                zarr_path=zarr_path,
                read_from_cache=read_from_cache,
                memory_size=memory_size,
                flush_cache=flush_cache,
                action_sampling_rate=action_sampling_rate,
                dataset_type='val',
                data_config=config,
                label_data_parser=label_parser)

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

    return (
        train_dataset,
        val_dataset,
        train_data_weights,
        val_data_weights
    )

def get_data_config_list(
        data_configs,
        datachunk_size=32,
        rank=0,
        dp_world_size=1,
        cam_mapping={'faceImg':'face_view',
                    'leftImg':'left_wrist_view',
                    'rightImg':'right_wrist_view'
                    },
        image_height=480,
        image_width=640,
        seed=42,
        flush_cache=False,
        return_full_traj=False,
        read_labeled_data=True,
        parse_tactile=False,
        trans_zh2en_url=None,
        num_workers=20,
        filter_angle_outliers=True,
        detect_motion=True,
        trim_stationary=False):
    """
    data_config: List of dict
    data_config = [
        {
            "path": "/x2robot/Data/2021-06-01-15-00-00", ## mandatory
            "instruction": "pick up the object",
            "obs_keys": ["face_view", "left_wrist_view", "right_wrist_view"],
            "action_keys": ["follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper", "follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper"]
            "cache": "/x2robot/Data/.cache/2021-06-01-15-00-00" 
            "dataset_group": "x2"
        }
    ]
    """

    for i,d in enumerate(data_configs):
        cache = d["cache"] if "cache" in d else "/x2robot/Data/.cache/"+"_".join(d["path"].split("/")[-2:])
        ## make cache file if not exists

        if i % dp_world_size == rank:
            if not os.path.exists(cache+"/0.zarr.zip") or flush_cache:
                zarr_path = cache+"/0.zarr.zip"
                print(f'{zarr_path} does not exist, creating')
                video_files = []
                for entry in glob.glob(f'{d["path"]}/*'):
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
                        num_workers=num_workers,
                        filter_angle_outliers=filter_angle_outliers,
                        detect_motion=detect_motion,
                        trim_stationary=d["trim_stationary"] if "trim_stationary" in d else trim_stationary)

    torch.distributed.barrier()

    dataload_config = []
    for i,d in enumerate(data_configs):
        cache = d["cache"] if "cache" in d else "/x2robot/Data/.cache/"+"_".join(d["path"].split("/")[-2:])        
        for cache_file in os.listdir(cache):
            if not cache_file.endswith(".zarr.zip"):
                continue
            cache_file = os.path.join(cache, cache_file)
            dataload_config.append({
                "path": d["path"],
                'cache_file': cache_file,
                'default_instruction': d["instruction"] if "instruction" in d else "",
                'action_horizon': 21,
                'action_history_length': 20,
                'image_horizon': 1,
                'image_history_length': 0,
                'right_padding': True,
                'left_padding': True,
                'obs_keys': d["obs_keys"] if "obs_keys" in d else ["face_view", "left_wrist_view", "right_wrist_view"],
                'action_keys':d['action_keys'] if "action_keys" in d else ["follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper", "follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper"],
                'train_val_split': 0.9,
                'split_seed': 42,
                "sample_ratio": d["sample_ratio"] if "sample_ratio" in d else 1.0,
                "action_sampling_rate": d["action_sampling_rate"] if "action_sampling_rate" in d else 1.0,
                "read_labeled_data": d["read_labeled_data"] if "read_labeled_data" in d else read_labeled_data,
                "dataset_group": d["dataset_group"] if "dataset_group" in d else "x2",
            })

    print(f"len of all cached datatset: {len(dataload_config)}",flush=True)
    random.shuffle(dataload_config)
    dataload_config = [dataload_config[i:i+datachunk_size] for i in range(0, len(dataload_config), datachunk_size)]
    print(f"len of data chunks: {len(dataload_config)}",flush=True)
    print(f"len of each data chunk: {[len(d) for d in dataload_config]}",flush=True)
    dataload_args = {
        "return_full_traj": return_full_traj,
        "trans_zh2en_url":trans_zh2en_url,
    }

    return dataload_config, dataload_args

from collections import defaultdict
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import av
import tqdm
import json
import random
import os
import torch
from datasets import Dataset, Features, Image, Sequence, Value


from safetensors.torch import save_file
from beartype import beartype

from x2robot_dataset.common.data_preprocessing import (
    process_videos,
    process_action,
    process_instruction,
    process_tactility,
    check_files,
    trim_stationary_ends,
    LabelData,
    _ACTION_KEY_EE_MAPPING_INV,
    _TAC_FILE_MAPPING,
    _HEAD_ACTION_MAPPING,
    _HEAD_ACTION_MAPPING_INV,
)

from types import SimpleNamespace

from .utils import (
    concatenate_episodes,
    get_default_encoding,
    check_repo_id,
)

from lerobot.common.datasets.utils import (
    flatten_dict,
    calculate_episode_data_index,
    hf_transform_to_torch
)

from lerobot.common.datasets.video_utils import VideoFrame
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    CODEBASE_VERSION
)

import logging
import glob


class LeRDataBuilder:
    def __init__(
        self,
        data_config:Dict
    ):
        self.config = SimpleNamespace(**data_config)
        self._feature_keys = [
            'index',
            'trim_index',
            'episode_index',
            'frame_index',
            'timestamp',
            # 'sample_name'
        ]

    def load_from_raw(
        self,
        video_files:List[str],
    ):
        def get_frame_count(video_path):
            with av.open(video_path, mode='r') as container:
                stream = container.streams.video[0]
                frame_count = stream.frames
                
                return frame_count

        report_file = os.path.join(os.path.dirname(video_files[0]), 'report.json')
        report_file = report_file if os.path.exists(report_file) else None

        valid_video_files, ee_osc_files = check_files(video_files,
                cam_list=self.config.cam_mapping.keys(),
                detect_phase_switch=self.config.filter_angle_outliers,
                detect_motion=self.config.detect_motion,
                report_file=report_file)
        
        ep_dicts = []
        for ep_idx, valid_path in enumerate(valid_video_files):
            ep_dict = {}

            robot_id, task_id, sample_name = valid_path.split('/')[-3:]

            filter_angle_outliers = self.config.filter_angle_outliers and (sample_name in ee_osc_files)
            inv_map = _ACTION_KEY_EE_MAPPING_INV | _HEAD_ACTION_MAPPING_INV if \
                self.config.parse_head_action else \
                _ACTION_KEY_EE_MAPPING_INV

            action_data = process_action(valid_path,
                                         action_key_mapping = inv_map,
                                         filter_angle_outliers=self.config.filter_angle_outliers)
            
            if self.config.trim_stationary:
                _, (trimmed_start, trimmed_end) = trim_stationary_ends(action_data, threshold=0.05)
            else:
                trimmed_start = 0
                trimmed_end = len(action_data[list(action_data.keys())[0]])
            
            for key, value in action_data.items():
                action_key = 'actions.' + key
                self._feature_keys.append(action_key)
                if value.ndim > 1:
                    ep_dict[action_key] = torch.tensor(value)
                else:
                    ep_dict[action_key] = torch.tensor(value).unsqueeze(1)

            first_action_key = list(action_data.keys())[0]

            if self.config.parse_tactile:
                tactile_data = process_tactility(valid_path)
                if tactile_data:
                    for key, value in tactile_data.items():
                        tac_key = 'tactile.' + key 
                        self._feature_keys.append(tac_key)
                        ep_dict[tac_key] = torch.tensor(value)

            mp4_files = glob.glob(os.path.join(valid_path, '*.mp4'))
            for fname in mp4_files:
                file_name = fname.split('/')[-1].split('.')[0]
                if file_name not in self.config.cam_mapping:
                    continue
                cam_name = self.config.cam_mapping[file_name]
                img_key = f"observations.{cam_name}"
                num_frames = get_frame_count(fname)
                assert num_frames == action_data[first_action_key].shape[0], \
                            f"Number of frames in video {fname} does not match number of actions"
                ep_dict[img_key] = [{"path": f"{fname}", "timestamp": i / self.config.fps} for i in range(num_frames)]
                self._feature_keys.append(img_key)
    
            ep_dict['trim_index'] = torch.cat(( \
                    torch.ones(trimmed_start, dtype=torch.int64), \
                    torch.zeros(trimmed_end - trimmed_start, dtype=torch.int64), \
                    torch.ones(num_frames - trimmed_end, dtype=torch.int64) \
                )
            )
            ep_dict['trim_index'][trimmed_start:trimmed_end] = 1
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / self.config.fps
            # ep_dict['sample_name'] = [robot_id + '_' + sample_name]*num_frames
            

            ep_dicts.append(ep_dict)
        
        data_dict = concatenate_episodes(ep_dicts)

        return data_dict

    def to_hf_dataset(
            self, 
            data_dict:Dict
    ):
        features = {}
        for key in self._feature_keys:
            if key.startswith('observations'):
                features[key] = VideoFrame()
            elif key.startswith('tactile'):
                # features[key] = Value(dtype="float32", id=None)
                features[key] = Sequence(
                    length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None)
                )
            elif key.startswith('actions'):
                # features[key] = Value(dtype="float32", id=None)
                features[key] = Sequence(
                    length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None)
                )


        features["trim_index"] = Value(dtype="int64", id=None)
        features["episode_index"] = Value(dtype="int64", id=None)
        features["frame_index"] = Value(dtype="int64", id=None)
        features["timestamp"] = Value(dtype="float32", id=None)
        features["index"] = Value(dtype="int64", id=None)
        # features["sample_name"] = Value(dtype="string", id=None)

        hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def save_meta_data(
        self,
        info: dict[str, Any], episode_data_index: dict[str, list], meta_data_dir: Path
    ):
        meta_data_dir.mkdir(parents=True, exist_ok=True)

        # save info
        info_path = meta_data_dir / "info.json"
        with open(str(info_path), "w") as f:
            json.dump(info, f, indent=4)
        
        data_config_path = meta_data_dir / "data_config.json"
        with open(str(data_config_path), "w") as f:
            json.dump(vars(self.config), f, indent=4)
        
        stats_path = meta_data_dir / "stats.safetensors"
        save_file(flatten_dict({}), stats_path)

        # save episode_data_index
        episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
        ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
        save_file(episode_data_index, ep_data_idx_path)
    
    def need_update_cache(self, data_config_path):
        if data_config_path.exists():
            with open(str(data_config_path), "r") as f:
                loaded_config = json.load(f)
            is_same_dict = (vars(self.config) == loaded_config)
        else:
            is_same_dict = False

        return is_same_dict
    

    @beartype
    def from_raw_to_lerobot_format(
        self,
        video_files:List[str],
        force_overwrite = False,
        repo_id:str|None = None,
        root_dir:Path|None = '/x2robot/Data/.cache/hf_datasets',
    ):
        def get_default_repo_id():
            default_repo_ids = []
            for video_file in video_files:
                robot_id, task_id, sample_name = video_file.split('/')[-3:]
                return robot_id + '/' + task_id
            if len(set(default_repo_ids)) == 1:
                return default_repo_ids[0]
            else:
                return None

        default_repo_id = get_default_repo_id()

        if default_repo_id is None and repo_id is None:
                raise ValueError("repo_id is required when multiple repo_ids are detected in the dataset")
        else:
            repo_id = repo_id or default_repo_id
        
        check_repo_id(repo_id)

        local_dir = root_dir / f"{repo_id}" # repo_id: 10001/pick_item_0819
        meta_data_dir = local_dir / "meta_data"
        videos_dir = Path('/')

        data_config_path = meta_data_dir / "data_config.json"
        if not force_overwrite and not self.need_update_cache(data_config_path):
            return repo_id, LeRobotDataset.from_preloaded(repo_id, videos_dir=videos_dir)
        
        logging.info(f"No cache found for {repo_id}. Generating cache...")


        random.seed(self.config.split_seed)
        random.shuffle(video_files)

        # def split_train_test(video_files, split_ratio):
        #     num_train = int(len(video_files) * split_ratio)
        #     random.seed(self.config.split_seed)
        #     random.shuffle(video_files)
    
        #     return video_files[:num_train], video_files[num_train:]

        # if self.config.train_test_split > 0.0 and self.config.train_test_split < 1.0:
        #     train_video_files, val_video_files = split_train_test(video_files, self.config.train_test_split)
        #     splits = ['train', 'val']
        # else:
        #     train_video_files = video_files
        #     splits = ['train']

        splits = ['train']
        for split in splits:
            data_dict= self.load_from_raw(video_files)

            hf_dataset = self.to_hf_dataset(data_dict)

            episode_data_index = calculate_episode_data_index(hf_dataset)
            
            info = {
                "codebase_version": CODEBASE_VERSION,
                "fps": self.config.fps,
                "video": True,
            }
            info["encoding"] = get_default_encoding()

            ler_dataset = LeRobotDataset.from_preloaded(
                repo_id=repo_id,
                hf_dataset=hf_dataset,
                episode_data_index=episode_data_index,
                info=info,
                videos_dir=videos_dir,
            )

            hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
            hf_dataset.save_to_disk(str(local_dir / split))

            self.save_meta_data(info, episode_data_index, meta_data_dir)

            return repo_id, ler_dataset
                
            
    
    
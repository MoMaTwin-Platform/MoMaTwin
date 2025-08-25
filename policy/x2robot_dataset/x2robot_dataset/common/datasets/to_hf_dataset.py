from collections import defaultdict
import re
from typing import List, Dict, Any
from pathlib import Path
from matplotlib.pylab import cond
from matplotlib.pyplot import cla
import numpy as np
import av
from regex import R
import tqdm
import json
import random
import os
import torch


from safetensors.torch import save_file
from beartype import beartype

from x2robot_dataset.common.data_preprocessing import (
    process_action,
    process_instruction,
    process_tactility,
    check_files,
    LabelData,
    _ACTION_KEY_FULL_MAPPING,
)

from types import SimpleNamespace

from .utils import (
    flatten_dict,
    check_repo_id,
    Annotation,
    join_dicts
)

import logging
import glob
from abc import ABC, abstractmethod

import cv2

import zarr
from zarr.sync import ThreadSynchronizer
from concurrent.futures import ThreadPoolExecutor
from x2robot_dataset.common.datasets.video_utils import decode_video_torchvision

CLASS_REGISTRY = {}
# {
#     "x2": X2DataBuilder,
#     "zarr": ZarrDataBuilder,
#     "hf": LazyDataBuilder
# }

def create_instance(
    data_config:Dict
):
    class_type = data_config["class_type"]
    if class_type in CLASS_REGISTRY:
        return CLASS_REGISTRY[class_type](data_config)
    else:
        raise ValueError(f"Unknown class type: {class_type}")

class MetaType(type):
    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        
        if name != 'DataBuilder':
            register_key = dct.get("REGISTER_KEY", name)
            CLASS_REGISTRY[register_key] = new_class
            # print(f"Registered class: {name} with key: {register_key}")
        
        return new_class

class DataBuilder(metaclass=MetaType):
    @abstractmethod
    def load_from_raw(self):
        pass

    @abstractmethod
    def to_hf_dataset(self):
        pass

    @abstractmethod
    def save_meta_data(self):
        pass

    @abstractmethod
    def need_update_cache(self):
        pass

    @abstractmethod
    def from_raw_to_videolazy_format(self):
        pass

    @abstractmethod
    def _get_default_repo_id(self):
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "load_from_raw")
            and callable(subclass.load_from_raw)
            and hasattr(subclass, "to_hf_dataset")
            and callable(subclass.to_hf_dataset)
            and hasattr(subclass, "save_meta_data")
            and callable(subclass.save_meta_data)
            and hasattr(subclass, "need_update_cache")
            and callable(subclass.need_update)
            and hasattr(subclass, "from_raw_to_videolazy_format")
            and callable(subclass.from_raw_to_videolazy_format)
            and hasattr(subclass, "_get_default_repo_id")
            and callable(subclass._get_default_repo_id)
        )

class LazyDataBuilder(DataBuilder):
    REGISTER_KEY = 'hf'

    def __init__(
        self,
        data_config:Dict
    ):
        self.config = SimpleNamespace(**data_config)
        self._feature_keys = [
            'sample_name',
            'length',
            'register_type',
        ]

    def load_from_raw(self):
        raise NotImplementedError

    def to_hf_dataset(
            self, 
            data_dict:Dict
    ):
        hf_dataset = Dataset.from_dict(data_dict)
        return hf_dataset

    def get_frame_count(self, video_path):
        with av.open(video_path, mode='r') as container:
            stream = container.streams.video[0]
            frame_count = stream.frames

            if stream.average_rate:
                fps = float(stream.average_rate)
            else:
                fps = float(stream.time_base.denominator) / float(stream.time_base.numerator)
    
        return frame_count, fps
                        

    def save_meta_data(
        self,
        info: dict[str, Any], meta_data_dir: Path
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
    
    def need_update_cache(self, data_config_path):
        
        def unify_data_type(data):
            if isinstance(data, list):
                return tuple(unify_data_type(i) for i in data)  # 列表转元组
            elif isinstance(data, dict):
                return {k: unify_data_type(v) for k, v in data.items()}  # 递归处理字典
            return data
        
        if data_config_path.exists():
            with open(str(data_config_path), "r") as f:
                loaded_config = json.load(f)
        else:
            return True
            
        is_same_dict = (
            unify_data_type(vars(self.config)) == unify_data_type(loaded_config)
        )
        
        return not is_same_dict

    @beartype
    def from_raw_to_videolazy_format(
        self,
        dataset_path:str,
        force_overwrite = False,
        save_meta_data = True,
        class_type:str = 'x2',
        split = 'train',
        num_threads:int = 32,
        root_dir:Path|None = '/x2robot/Data/.cache/hf_datasets',
    ):
        
        repo_id = self._get_default_repo_id(dataset_path)
        check_repo_id(repo_id)

        if class_type == 'hf':
            return {
                'repo_id':repo_id,
                'hf_dataset': None,
                'info': None,
            }

        local_dir = root_dir / f"{repo_id}" # repo_id: 10001/pick_item_0819
        meta_data_dir = local_dir / "meta_data"

        data_config_path = meta_data_dir / "data_config.json"
        if not force_overwrite and not self.need_update_cache(data_config_path):
            logging.info(f"Cache found for {repo_id}. Loading cache...")
            return {
                'repo_id':repo_id,
                'hf_dataset': None,
                'info': None,
            }
        
        # if class_type == 'x2':
        #     data_dict= self.load_from_raw(dataset_path=dataset_path)
        if class_type in ['x2', 'image', 'video']:
            # check if the dataset_path is a json file
            if dataset_path.endswith('.json'):
                ### TODO: 这样根据类型决定不好
                d = json.load(open(dataset_path, 'r'))
                if isinstance(d, dict):
                    data_dict= self.load_from_json(dataset_path)
                else:
                    data_dict= self.load_from_raw(dataset_path=dataset_path)
            else:
                data_dict= self.load_from_raw(dataset_path=dataset_path)
        elif class_type == 'zarr':
            data_dict= self.load_from_raw(zarr_path=dataset_path, root=local_dir, num_threads=num_threads)
        else:
            raise ValueError(f"Unknown class type: {class_type}")

        hf_dataset = self.to_hf_dataset(data_dict)
        
        if class_type in ['zarr', 'x2', 'video']:
            info = {
                "fps": hf_dataset['fps'][0] if 'fps' in hf_dataset else self.config.fps,
                "video": True,
                "num_frames": sum(hf_dataset['length']),
                "num_episodes": len(hf_dataset['length']),
            }
        elif class_type == 'image':
            info = {
                "video": False,
                "num_images": sum([len(x) for x in hf_dataset['image']]),
                "num_episodes": len(hf_dataset['image']),
            }
        else:
            raise ValueError(f"Unknown class type: {class_type}")
        
        if save_meta_data: #按理说只需要一个rank保存，为什么所有rank都要搞一遍
            logging.info(f"No cache found for {repo_id}. Overwrite cache...")
            hf_dataset.save_to_disk(str(local_dir / split))
            self.save_meta_data(info, meta_data_dir)

        return {
            'repo_id':repo_id,
            'hf_dataset': hf_dataset,
            'info': info,
        }

    def _get_default_repo_id(self, dataset_path:str):
        return dataset_path
    


class X2DataBuilder(LazyDataBuilder):
    REGISTER_KEY = 'x2'
    ANNOTATION_SINGLE = "single_frame"      # 单帧描述
    ANNOTATION_INTERVAL = "interval"        # 间隔帧描述
    ANNOTATION_CONTINUOUS = "continuous"    # 连续片段描述

    def __init__(self, data_config:Dict):
        super().__init__(data_config)
        self._feature_keys.extend([
            'trim_start',
            'trim_end',
            'fps',
        ])
        self.robot_id = None

    def load_from_raw(
        self,
        dataset_path:str,
    ):
        if dataset_path.endswith('.json'):
            ## 以json的形式指定需要的episodes：["episode_dir_1", "episode_dir_2", ...]
            episode_paths = json.load(open(dataset_path, 'r'))
            valid_video_files = []
            for episode_path in episode_paths:
                if os.path.isdir(episode_path):
                    has_mp4_files = len(glob.glob(f'{episode_path}/*.mp4')) > 0
                    if has_mp4_files:
                        valid_video_files.append(episode_path)
            random.seed(self.config.split_seed)
            random.shuffle(valid_video_files)
            valid_video_files, ee_osc_files = check_files(
                valid_video_files,
                cam_list=self.config.cam_mapping.keys(),
                detect_phase_switch=self.config.filter_angle_outliers,
                detect_motion=self.config.detect_motion,
            )
            print(f"Found {len(valid_video_files)} episodes in {dataset_path}", flush=True)
        else:
            video_files = []
            for entry in glob.glob(f'{dataset_path}/*'):
                if os.path.isdir(entry):
                    has_mp4_files = len(glob.glob(f'{entry}/*.mp4')) > 0
                    if has_mp4_files:
                        video_files.append(entry)

            random.seed(self.config.split_seed)
            random.shuffle(video_files)

            report_file = os.path.join(os.path.dirname(video_files[0]), 'report.json')
            report_file = report_file if os.path.exists(report_file) else None

            valid_video_files, ee_osc_files = check_files(
                video_files,
                cam_list=self.config.cam_mapping.keys(),
                detect_phase_switch=self.config.filter_angle_outliers,
                detect_motion=self.config.detect_motion,
                report_file=report_file
            )
        
        ep_dicts = []
        for ep_idx, valid_path in enumerate(valid_video_files):
            ep_dict = {}
            valid = True
            robot_id, task_id, sample_name = [s for s in valid_path.split('/')[-3:] if s != ""]

            filter_angle_outliers = self.config.filter_angle_outliers and (sample_name in ee_osc_files)
            inv_map = {}
            inv_map = {_ACTION_KEY_FULL_MAPPING[k]:k for k in self.config.action_keys}
            action_data = process_action(valid_path,
                                         action_key_mapping = inv_map,
                                         filter_angle_outliers=filter_angle_outliers)
                                        #  filter_angle_outliers=self.config.filter_angle_outliers)
            
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
                        tac_key = 'tactiles.' + key 
                        self._feature_keys.append(tac_key)
                        ep_dict[tac_key] = torch.tensor(value)

            # if self.config.parse_depth_image:
            #     # load json file
            #     # print(sample_name)
            #     if not hasattr(self, 'depth_data'):
            #         print('load depth data')
            #         depth_json_path = '/x2robot/caichuang/extra_codes/depth/Video-Depth-Anything/new_data.json'
            #         self.depth_data = json.load(open(depth_json_path))
            #     depth_data = self.depth_data
            #     left_depth_video_full_path = depth_data[f'{sample_name}']['left_depth_video_path']
            #     right_depth_video_full_path = depth_data[f'{sample_name}']['right_depth_video_path']

            #     left_depth_video_path = os.path.join(valid_path, left_depth_video_full_path)
            #     right_depth_video_path = os.path.join(valid_path, right_depth_video_full_path)

            #     # Load mp4 file as numpy array(convert to float32)
            #     left_depth_video = decode_video_torchvision(left_depth_video_path).astype(np.float32)
            #     right_depth_video = decode_video_torchvision(right_depth_video_path).astype(np.float32)
            #     # print(left_depth_video.shape, right_depth_video.shape)

            #     ep_dict['left_depth_video'] = left_depth_video
            #     ep_dict['right_depth_video'] = right_depth_video
            #     self._feature_keys.extend(['left_depth_video', 'right_depth_video'])
                # print(left_depth_video.shape, right_depth_video.shape)

            mp4_files = glob.glob(os.path.join(valid_path, '*.mp4'))
            for fname in mp4_files:
                file_name = fname.split('/')[-1].split('.')[0]
                if file_name not in self.config.cam_mapping:
                    continue
                cam_name = self.config.cam_mapping[file_name]
                img_key = f"observations.{cam_name}"
                num_frames, fps = self.get_frame_count(fname)
                if num_frames != action_data[first_action_key].shape[0]:
                    print(f"Number of frames in video {fname} does not match number of actions, {num_frames} != {action_data[first_action_key].shape[0]}")
                    valid = False

                # assert num_frames == action_data[first_action_key].shape[0], \
                #             f"Number of frames in video {fname} does not match number of actions, {num_frames} != {action_data[first_action_key].shape[0]}"
                ep_dict[img_key] = fname
                self._feature_keys.append(img_key)


            instruction_path = self.config.instruction_path
            if instruction_path is not None:
                instruction_path = instruction_path.format(robot_id=robot_id, topic_id=task_id, uid=sample_name)
                if os.path.exists(instruction_path):
                    instruction_info = json.load(open(instruction_path, 'r'))
                    for key, value in instruction_info.items():
                        ep_dict[key] = value
                    self._feature_keys.extend(instruction_info.keys())
        
            ep_dict['fps'] = fps
            ep_dict['trim_start'] = trimmed_start
            ep_dict['trim_end'] = trimmed_end
            ep_dict['sample_name'] = robot_id + '_' + sample_name
            ep_dict['name'] = sample_name
            ep_dict['length'] = action_data[first_action_key].shape[0] # number of frames, no trimming
            ep_dict['register_type'] = X2DataBuilder.REGISTER_KEY
            if valid:
                ep_dicts.append(ep_dict)
    
        data_dict = {key: [ep_dict[key] if key in ep_dict else None for ep_dict in ep_dicts] for key in self._feature_keys}

        return data_dict
    
    def load_from_json(self, json_file) -> dict:
        '''
        Load data from a json file
        {
            'x2_data': {
                'robot_data': x2_robot_data_folder,
                'conditions': [
                    {
                        'sample_name': 'sample_name1',
                        'image': {
                            'leftImg': {
                                'frame_indices': [13, 20],
                                'fig': ['image1_path', 'image2_path'],
                            },
                            'faceImg': {
                                ...
                            },
                            ...
                        }
                        'video': {
                            'leftImg': 'video_path',
                            'faceImg': 'video_path',
                            ...
                        },
                        'text': [
                            {
                                'annotation': 'Instruction text',
                                'annotation_type': 'single_frame',
                                'frame_indices': [30]
                            },
                            {
                                'annotation': 'Instruction text',
                                'annotation_type': 'interval',
                                'frame_indices': [30, 48, 60]
                            },
                            {
                                'annotation': 'Instruction text',
                                'annotation_type': 'continuous',
                                'frame_indices': [30, 90]
                            }
                            ...
                        ]
                    },
                    {
                        'sample_name': 'sample_name2',
                        ...
                    },
                    ...
                ]
            }
        }
        '''
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if 'x2_data' not in json_data:
            raise ValueError("The json file should contain a key 'x2_data'")
        
        x2_data = json_data['x2_data']
        if 'robot_data' not in x2_data:
            raise ValueError("The json file should contain a key 'robot_data'")
        
        robot_data = x2_data['robot_data']
        if not os.path.exists(robot_data):
            raise FileNotFoundError(f"Robot data folder not found: {robot_data}")
        
        data_dict = self.load_from_raw(robot_data)

        if 'conditions' not in x2_data:
            return data_dict

        def get_cam_filename():
            inverse_cam_mapping = {v:k for k,v in self.config.cam_mapping.items()}
            cam_keys = []
            for key in data_dict.keys():
                if key.startswith('observations'):
                    file_name = key.split('.')[-1]
                    if file_name in inverse_cam_mapping:
                        cam_name = inverse_cam_mapping[file_name]
                        cam_keys.append(cam_name)
            return cam_keys
        cam_names = get_cam_filename()

        ep_dicts = []
        for cond_data in x2_data['conditions']:
            ep_dict = {}
            ep_dict['sample_name'] = cond_data['sample_name']
            for key in ['image', 'video']:
                for filename in cam_names:
                    cam_key = self.config.cam_mapping[filename]
                    cond_key = 'conditions.' + key + '.' + cam_key

                    if key not in cond_data or filename not in cond_data[key]:
                        ep_dict[cond_key] = None
                    else:
                        cam_data = cond_data[key][filename]
                        ep_dict[cond_key] = cam_data
                    self._feature_keys.append(cond_key)
            
            text_key = 'conditions.text'
            if 'text' not in cond_data:
                ep_dict[text_key] = None
            else:
                ep_dict[text_key] = cond_data['text']
            self._feature_keys.append(text_key)
            
            ep_dicts.append(ep_dict)
        
        ep_dicts = {key: [ep_dict[key] for ep_dict in ep_dicts] for key in ep_dicts[0].keys()}

        data_dict = join_dicts(data_dict, ep_dicts, join_key='sample_name')
        
        return data_dict
    
    
    def _get_default_repo_id(self, dataset_path:str):
        # return '/'.join(dataset_path.split('/')[-2:])
        if dataset_path.endswith('.json'):
            return '/'.join(os.path.abspath(dataset_path).split('/')[-3:-1])
        else:
            return '/'.join(os.path.abspath(dataset_path).split('/')[-2:])


class ZarrDataBuilder(LazyDataBuilder):
    REGISTER_KEY = 'zarr'
    def __init__(self, data_config:Dict):
        super().__init__(data_config)
        self._feature_keys.extend([
            'trim_start',
            'trim_end',
        ])
    
    def load_from_raw(
        self,
        zarr_path: str,
        root: Path,
        num_threads: int = 16
    ):
        def video_to_mp4(video_array, output_file, fps=30):
            frames, height, width, channels = video_array.shape

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            # Write each frame to the video
            for frame in video_array:
                # video_writer.write(frame)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            video_writer.release()


        def load_and_index_json(file_path):
            dir_name = os.path.dirname(file_path)
            inst_file = os.path.join(dir_name, 'instructions.json')
            
            uid_index = {}
            if not os.path.exists(inst_file):
                return uid_index
            
            with open(inst_file, 'r', encoding='utf-8') as file:
                for line in file:
                    record = json.loads(line.strip())
                    uid_index[record["uid"]] = record["instruction"][0]
            return uid_index
        
        def process_episode(i, starts, ends, action_start, replay_buffer, sample_names, video_path, inst_dict):
            ep_dict = {}
            for key in replay_buffer.data.keys():
                if key.startswith('follow'):
                    st, ed = action_start + starts[i], action_start + ends[i]
                    action_key = 'actions.' + key
                    ep_dict[action_key] = replay_buffer.data[key][st:ed]
                    self._feature_keys.append(action_key)
                elif key == 'sample_names':
                    ep_dict['sample_name'] = sample_names[i]
                elif key.endswith('view'):
                    save_path = str(video_path / f'{key}_{i}.mp4')
                    if not os.path.exists(save_path):                    
                        video_array = replay_buffer.data[key][starts[i]:ends[i]]
                        video_to_mp4(video_array, save_path)
                    
                    mapped_key = self.config.cam_mapping[key] if key in self.config.cam_mapping else key
                    img_key = 'observations.' + mapped_key
                    ep_dict[img_key] = save_path
                    self._feature_keys.append(img_key)
                else:
                    pass

            if inst_dict:
                inst_key = 'default_instruction'
                ep_dict[inst_key] = inst_dict[ep_dict['sample_name']]
                self._feature_keys.append(inst_key)
            ep_dict['trim_start'] = 0
            ep_dict['trim_end'] = ends[i] - starts[i]
            ep_dict['length'] = ends[i] - starts[i]
            ep_dict['register_type'] = ZarrDataBuilder.REGISTER_KEY

            return ep_dict

        def process_zarr_episodes_in_parallel(zarr_path, root, num_threads=num_threads):
            store = zarr.ZipStore(zarr_path, mode='r')
            replay_buffer = zarr.open(store, mode='r', synchronizer=ThreadSynchronizer())

            ends = list(replay_buffer.meta['episode_ends'])
            sample_names = replay_buffer.data['sample_names'][-len(ends):]
            action_start = ends[0]

            ends = ends - ends[0]
            starts, ends = ends[:-1], ends[1:]

            inst_dict = load_and_index_json(zarr_path)

            video_path = Path(root) / 'videos'
            video_path.mkdir(parents=True, exist_ok=True)

            ep_dicts = []

            # Use ThreadPoolExecutor for multithreading
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(
                        process_episode, i, starts, ends, action_start, replay_buffer, sample_names, video_path, inst_dict
                    )
                    for i in range(len(starts))
                ]

                # Use tqdm to track progress
                for future in tqdm.tqdm(futures, desc='Processing zarr episodes'):
                    ep_dicts.append(future.result())

            return ep_dicts
        
        ep_dicts = process_zarr_episodes_in_parallel(
                zarr_path,
                root,
                num_threads=num_threads
            )
        data_dict = {key: [ep_dict[key] for ep_dict in ep_dicts] for key in self._feature_keys}

        return data_dict

    def _get_default_repo_id(self, zarr_path):
        try:
            dataset_id = zarr_path.split('/')[-1].split('.')[0] # 0.zarr.zip
            return zarr_path.split('/')[-2] + '/' + dataset_id
        except:
            return None
    

from PIL import Image


class ImageDataBuilder(LazyDataBuilder):
    REGISTER_KEY = 'image'
    ANNOTATION_SINGLE = "single_image"
    ANNOTATION_TRANSFORM = "transformation"

    def __init__(self, data_config:Dict):
        super().__init__(data_config)
        self._feature_keys.extend([
            'image',
            'text',
            'annotation_type'
        ])

    def load_from_raw(self, dataset_path: str) -> Dict[str, List]:
        """
        从原始数据路径加载图像数据集，目录结构：
        folder1/
            - image1.jpg
            - image2.jpg
            - metadata.json
        或者
        data_folder/
            folder1/
                image1.jpg
                image2.jpg
                metadata.json
            folder2/
                image1.jpg
                image2.jpg
                metadata.json
        
        metadata.json 格式示例：
        {
            "text": {
                "image1": "描述图片1内容",
                "image2": "描述图片2内容"
            }
        }
        """
        def get_images_and_metadata(folder: str):
            """ 获取文件夹内的图片和 metadata.json """
            image_files = glob.glob(f'{folder}/*.jpg') + glob.glob(f'{folder}/*.png')
            # image_files 保存为绝对路劲
            image_files = [os.path.abspath(f) for f in image_files]

            meta_file = os.path.join(folder, 'metadata.json')
            annotations = {}
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)['text']

            return image_files, annotations
        
        def _process_single_folder(image_files: List[str], annotations: Dict[str, str], folder_name: str) -> Dict:
            """ 处理单个文件夹中的多张图片、分辨率和文本 """

            ep_dicts = []
            for img_path in image_files:
                sample_name = img_path.split('/')[-1].split('.')[0]
                ep_dicts.append({
                    "sample_name": sample_name,
                    'length': 1,
                    'register_type': ImageDataBuilder.REGISTER_KEY,
                    "image": [img_path],
                    "text": annotations.get(sample_name, "No ANNOTATION available"),
                    "annotation_type": self.ANNOTATION_SINGLE,
                })
                            
            return ep_dicts

        ep_dicts = []

        # 检测 dataset_path 目录结构
        first_level_folders = [f for f in glob.glob(f'{dataset_path}/*') if os.path.isdir(f)]
        has_nested_folders = any(os.path.isdir(f) for f in first_level_folders)

        if not has_nested_folders:
            # **单层文件夹**
            image_files, annotations = get_images_and_metadata(dataset_path)
            if image_files:
                ep_dicts.extend(_process_single_folder(image_files, annotations, dataset_path))

        else:
            for sample_folder in first_level_folders:
                image_files, annotations = get_images_and_metadata(sample_folder)
                if image_files:  # **仅处理包含图片的目录**
                    ep_dicts.extend(_process_single_folder(image_files, annotations, sample_folder))

        # 组织数据
        data_dict = {format_key: [ep_dict[format_key] for ep_dict in ep_dicts] for format_key in self._feature_keys}

        return data_dict

    def load_from_json(self, json_file):
        '''
        Load data from a json file
        {
            'image_data':[
                {
                    'sample_name': 'sample_name1',
                    'image': ['image1_path','image2_path'],
                    'text': ['ANNOTATION_text1','ANNOTATION_text2'],
                    'annotation_type': 'single_image'
                },
                {
                    'sample_name': 'sample_name2',
                    'image': ['image1_path','image2_path'],
                    'text': ['ANNOTATION_text'],
                    'annotation_type': 'transformation'
                },
                ...
            ]
        }
        '''
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # check if the json file is in the correct format
        if 'image_data' not in json_data:
            raise ValueError("The json file should contain a key 'image_data'")
    
        # check keys in the image_data, if not matching self._feature_keys, raise error
        ep_dicts = json_data['image_data']
        for data in ep_dicts:
            data['register_type'] = ImageDataBuilder.REGISTER_KEY
            data['length'] = len(data['image']) if 'image' in data else 0
            for key in self._feature_keys:
                if key not in data:
                    raise ValueError(f"Key {key} is missing in the json file")
        
        data_dict = {key: [data[key] for data in ep_dicts] for key in self._feature_keys}
    
        return data_dict

    def _get_default_repo_id(self, dataset_path: str):
        # check if the dataset_path is a json file
        if dataset_path.endswith('.json'):
            return '/'.join(os.path.abspath(dataset_path).split('/')[-3:-1])
        else:
            return '/'.join(os.path.abspath(dataset_path).split('/')[-2:])
        

class VideoDataBuilder(LazyDataBuilder):
    REGISTER_KEY = 'video'
    ANNOTATION_SINGLE = "single_frame"             # 单帧描述
    ANNOTATION_INTERVAL = "interval"        # 间隔帧描述
    ANNOTATION_CONTINUOUS = "continuous"    # 连续片段描述

    def __init__(self, data_config: Dict):
        super().__init__(data_config)
        self._feature_keys.extend([
            'video',
            'text',
            'fps'
        ])

    def load_from_raw(self, dataset_path: str) -> Dict[str, List]:
        """
        从原始数据路径加载视频数据集，目录结构：
        dataset_path/
            - video1.mp4
            - video2.mp4
            - metadata.json
        
        metadata.json 格式示例：
        {
            "text": {
                "video1": {
                    "type": "single_frame",
                    "annotation":  "描述第30帧的内容",
                    frame_indices: [30]
                },
                "video2": {
                    "type": "interval",
                    "annotation": "27,30,60帧的描述",
                    frame_indices: [27, 30, 60]
                    }
                },
                "video3": {
                    "type": "continuous",
                    "annotation": 描述30-90帧的内容",
                    frame_indices: [30, 90]
                }
            }
        }
        """

        # 获取所有视频文件
        video_files = glob.glob(f'{dataset_path}/*.mp4') + \
                     glob.glob(f'{dataset_path}/*.avi') + \
                     glob.glob(f'{dataset_path}/*.mov')
        video_files = [os.path.abspath(f) for f in video_files]

        # 读取 metadata
        meta_file = os.path.join(dataset_path, 'metadata.json')
        annotations = {}
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                annotations = metadata.get('text', {})

        ep_dicts = []
        for video_path in video_files:
            sample_name = video_path.split('/')[-1].split('.')[0]
            # duration, fps = get_video_info(video_path)

            if sample_name in annotations:
                video_annotation = annotations[sample_name]
                text = [{
                            'annotation_type': video_annotation['type'],
                            'frame_indices': video_annotation['frame_indices'],
                            'annotation': video_annotation['annotation']
                        }]
            else:
                text = []

            frame_count, fps = self.get_frame_count(video_path)
                    
            ep_dicts.append({
                "sample_name": sample_name,
                'fps': fps,
                'length': 1,
                'register_type': VideoDataBuilder.REGISTER_KEY,
                "video": [video_path],
                "text": text
            })


        data_dict = {format_key: [ep_dict[format_key] for ep_dict in ep_dicts] for format_key in self._feature_keys}
        return data_dict

    def load_from_json(self, json_file):
        '''
        Load data from a json file
        {
            'video_data':[
                {
                    'sample_name': 'sample_name1',
                    'video': ['video_path'],
                    'text': {
                        'annnotation':'Frame 30 description',
                        'annotation_type': 'single_frame',
                        ''frame_indices': [30]
                    }
                },
                {
                    'sample_name': 'sample_name2',
                    'video': ['video_path'],
                    'text': {
                        'annotation': 'Description for frames 0, 30, 60',
                        'annotation_type': 'interval',
                        'frame_indices': [0, 30, 60]
                    }
                },
                {
                    'sample_name': 'sample_name3',
                    'video': ['video_path'],
                    'text': {
                        'annotation': 'Description for continuous frames 30-90',
                        'annotation_type': 'continuous',
                        'frame_indices': [30, 90]
                    }
                },
                {
                    'sample_name': 'sample_name4',
                    'video': ['view1_path', 'view2_path', 'view3_path'],
                    'text': {
                        'annotation': 'Description for continuous frames 30-90',
                        'annotation_type': 'continuous',
                        'frame_indices': [30, 90]
                    }
                },
            ]
        }
        '''
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        if 'video_data' not in json_data:
            raise ValueError("The json file should contain a key 'video_data'")
        
        ep_dicts = json_data['video_data']
        for data in ep_dicts:
            data['register_type'] = VideoDataBuilder.REGISTER_KEY
            data['length'] = 1
            
            # 验证必需的字段
            for key in self._feature_keys:
                if key not in data:
                    raise ValueError(f"Key {key} is missing in the json file")
            
            # 验证annotation_type是否为有效值
            if data['annotation_type'] not in [self.ANNOTATION_SINGLE, 
                                          self.ANNOTATION_INTERVAL,
                                          self.ANNOTATION_CONTINUOUS]:
                raise ValueError(f"Invalid annotation_type: {data['annotation_type']}")
            
            # 验证frame_indices格式是否正确
            if data['annotation_type'] == self.ANNOTATION_SINGLE and len(data['frame_indices']) != 1:
                raise ValueError(f"Single frame annotation should have exactly one frame index")
            elif data['annotation_type'] == self.ANNOTATION_CONTINUOUS and len(data['frame_indices']) != 2:
                raise ValueError(f"Continuous annotation should have start and end frame indices")
            
            # 验证视频文件是否存在
            for video_path in data['video']:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                _, fps = self.get_frame_count(video_path)
                if 'fps' not in data:
                    data['fps'] = fps
            
            for key in self._feature_keys:
                if key not in data:
                    raise ValueError(f"Key {key} is missing in the json file")

        
        data_dict = {key: [data[key] for data in ep_dicts] for key in self._feature_keys}
    
        return data_dict

    def _get_default_repo_id(self, dataset_path: str):
        if dataset_path.endswith('.json'):
            return '/'.join(os.path.abspath(dataset_path).split('/')[-3:-1])
        else:
            return '/'.join(os.path.abspath(dataset_path).split('/')[-2:])



                
            
    
    
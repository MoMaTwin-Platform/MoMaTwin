from curses import raw
import re
# from anyio import key
import torch
from pathlib import Path
import os
import datasets
import time
from x2robot_dataset.common.datasets.utils import (
    split_data,
    load_hf_dataset,
    load_stats,
    load_info,
    load_config,
    load_videos,
)

from x2robot_dataset.common.datasets.to_hf_dataset import (
    VideoDataBuilder,
    X2DataBuilder
)

from x2robot_dataset.common.datasets.video_utils import decode_video_torchvision
import bisect
from copy import deepcopy
import numpy as np
from collections import defaultdict

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None

from PIL import Image

class LazyDatasetBase(torch.utils.data.Dataset):
    def __init__(self, 
                 repo: dict|str = None,
                 root: Path | None = DATA_DIR,
                 split: str = "train",
                 ) -> None:
        super().__init__()
        self.repo_id = repo['repo_id'] \
                        if isinstance(repo, dict) and 'repo_id' in repo \
                        else repo
        self.root = root
        self.split = split

        # 满足if条件，repo["hf_dataset"]不为None, 说明hf_dataset被重新生成了，所以直接使用,
        # 一般对应参数设置为： --force_overwrite True 或者 --force_overwrite False但config有变化
        # 满足else条件，说明hf_dataset没有变化，从磁盘加载
        # 一般对应参数设置为： --force_overwrite False且config没有变化
        # 见from_raw_to_videolazy_format@to_hf_dataset
        self.hf_dataset = split_data(repo['hf_dataset'], split) \
                            if isinstance(repo, dict) \
                                and 'hf_dataset' in repo \
                                and repo['hf_dataset'] is not None \
                            else load_hf_dataset(self.repo_id, root, split)

        self.info = repo['info'] \
                        if isinstance(repo, dict) \
                            and 'info' in repo \
                            and repo['info'] is not None \
                        else load_info(self.repo_id, root)

        self.config = repo['config'] \
                        if isinstance(repo, dict) \
                            and 'config' in repo \
                            and repo['config'] is not None \
                        else load_config(self.repo_id, root)

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self.repo_id
    
    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset)

    @property
    def frame_count_list(self) -> list[int]:
        """List of frame counts for each episode"""
        return [item['length'] for item in self.hf_dataset]

    @property
    def num_frames(self) -> int:
        """Total number of frames in the dataset"""
        return sum(self.frame_count_list)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        '''Returns a dictionary containing the data for the given index.'''
        pass
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f")"
        )

class VideoLazyDataset(LazyDatasetBase):
    def __init__(self, 
                 repo: dict|str = None,
                 root: Path | None = DATA_DIR,
                 split: str = "train",
                 ) -> None:
        super().__init__(repo, root, split)

        self._actual_num_frames = None  # Add cache for sampled frame count
        self.frame_count_list_results = None
        self.get_frame_count_list()

        # 耗时极大
        # print(f"Dataset {split} - Total episodes: {len(self.hf_dataset)}")
        # print(f"Dataset {split} - Frame count list: {len(self.frame_count_list)}")
        # print(f"Dataset {split} - Total frames: {self.num_frames}")
    
    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]
    
    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if key.startswith("observations"):
                keys.append(key)
        return keys
    
    @property
    def condition_keys(self) -> list[str]:
        """Keys to access condition data."""
        keys = []
        for key, feats in self.features.items():
            if key.startswith("conditions"):
                keys.append(key)
        return keys

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset)
    
    def get_frame_count_list(self):
        # self.frame_count_list_results = self.hf_dataset["length"]
        counts = []
        for item in self.hf_dataset:
            length = item['length']
            if self.config['sample_rate'] < 1:
                # Calculate the actual sampled indices
                sample_indices = np.linspace(0, length-1, 
                                        int(length * self.config['sample_rate']),
                                        dtype=int)
                if 'trim_start' in item and 'trim_end' in item:
                    # Find the actual number of frames after sampling
                    start_idx = np.searchsorted(sample_indices, item['trim_start'])
                    end_idx = np.searchsorted(sample_indices, item['trim_end'], side='right') - 1
                    length = end_idx - start_idx + 1
                else:
                    length = len(sample_indices)
            elif 'trim_start' in item and 'trim_end' in item:
                length = item['trim_end'] - item['trim_start']
            counts.append(length)
        self.frame_count_list_results = counts

    @property
    def frame_count_list(self) -> list[int]:
        """List of frame counts for each episode after applying sample rate."""
        if self.frame_count_list_results is None:
            self.get_frame_count_list()
        return self.frame_count_list_results

    @property
    def num_frames(self) -> int:
        """Total number of frames in the dataset after applying sample rate."""
        if self._actual_num_frames is None:
            self._actual_num_frames = sum(self.frame_count_list)
        return self._actual_num_frames

    def __getitem__(self, idx):
        '''Returns a dictionary containing the data for the given index.'''
        
        item = deepcopy(self.hf_dataset[idx])
        
        if self.config['sample_rate'] >= 1 or self.config['sample_rate'] <= 0:
            self.config['sample_rate'] = 1
        # Calculate sampling indices
        # if self.config['sample_rate'] < 1:
        sample_indices = np.linspace(0, item['trim_end']-item['trim_start']-1,   
                                        int((item['trim_end']-item['trim_start']) * self.config['sample_rate']),
                                        dtype=int)
        # use numpy array to sample the data
        raw_to_sampling_indices =  {i:idx for idx, i in enumerate(sample_indices)}
        
        # Sample camera views
        for key in self.camera_keys:
            video_path = item[key]
            video_path = str(video_path) if isinstance(video_path, Path) else video_path
            frames = decode_video_torchvision(video_path)
            frames = frames[item['trim_start']:item['trim_end']]
            item[key] = frames[sample_indices]

        for key in self.condition_keys:
            part = key.split('.')[1]
            if part.startswith('video'):
                video_path = item[key]
                if video_path is None:
                    continue

                frames = decode_video_torchvision(video_path)
                frames = frames[item['trim_start']:item['trim_end']]
                item[key] = frames[sample_indices]
            elif part.startswith('image'):
                img = item[key] # conditions.image.left_wrist_view
                if img is None:
                    continue
                if img['fig'] is None:
                    continue
                                
                images_numpy = []
                for img_path in img['fig']:
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert("RGB")  # 确保是 3 通道
                            img_array = np.array(img, dtype=np.uint8)  # 转换为 numpy
                            images_numpy.append(img_array)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}", flush=True)
                        images_numpy.append(None)
                item[key]['fig'] = images_numpy
                item[key]['frame_indices'] = [raw_to_sampling_indices[i] for i in item[key]['frame_indices']]
            elif part.startswith('text'):
                annotations = item[key]
                if annotations is None:
                    continue
                
                frame_indices, texts = [],[]
                for annotation in annotations:
                    frame_idx = [raw_to_sampling_indices[i] for i in annotation["frame_indices"]]
                    if annotation['annotation_type'] == X2DataBuilder.ANNOTATION_CONTINUOUS:
                        st,ed = frame_idx
                        frame_indices.extend(list(range(st, ed)))
                        texts.extend([annotation["annotation"]]*(ed-st))
                    else:
                        frame_indices.extend(frame_idx)
                        texts.extend([annotation["annotation"]]*len(frame_idx))
                
                item[key] = {'frame_indices': frame_indices, 'annotations': texts}                    
            
        # Sample actions with same indices
        for key in item.keys():
            if key.startswith('actions.'):
                actions = np.array(item[key])  # Convert to numpy array first
                item[key] = actions[item['trim_start']:item['trim_end']]
                item[key] = item[key][sample_indices]
            if key.startswith('tactiles.'):
                tactile = np.array(item[key])
                item[key] = tactile[item['trim_start']:item['trim_end']]
                item[key] = item[key][sample_indices]

        # Update length
        item['length'] = item['trim_end'] - item['trim_start']
        item['trim_start'] = 0
        item['trim_end'] = item['length']
        # else:
        #     # Load videos normally if no sampling is needed
        #     for key in self.camera_keys:
        #         video_path = item[key]
        #         video_path = str(video_path) if isinstance(video_path, Path) else video_path
        #         item[key] = decode_video_torchvision(video_path)
        #         item[key] = item[key][item['trim_start']:item['trim_end']]
        #     for key in item.keys():
        #         if key.startswith('actions.'):
        #             item[key] = np.array(item[key])[item['trim_start']:item['trim_end']]
        #         if key.startswith('tactiles.'):
        #             item[key] = np.array(item[key])[item['trim_start']:item['trim_end']]
            
        #     if 'trim_start' in item and 'trim_end' in item:
        #         item['length'] = item['trim_end'] - item['trim_start']
        #         item['trim_start'] = 0
        #         item['trim_end'] = item['length']

            # # if self.config.parse_depth_image:
            # # load json file
            # # print(sample_name)
            # sample_name_ = item['name']
            # if not hasattr(self, 'depth_data'):
            #     print('load depth data')
            #     depth_json_path = '/x2robot/caichuang/extra_codes/depth/Video-Depth-Anything/new_data.json'
            #     self.depth_data = json.load(open(depth_json_path))
            # depth_data = self.depth_data
            # left_depth_video_full_path = depth_data[f'{sample_name_}']['left_depth_video_path']
            # right_depth_video_full_path = depth_data[f'{sample_name_}']['right_depth_video_path']

            # # Load mp4 file as numpy array(convert to float32)
            # left_depth_video = decode_video_torchvision(left_depth_video_full_path).astype(np.float32)
            # right_depth_video = decode_video_torchvision(right_depth_video_full_path).astype(np.float32)
            # # print(left_depth_video.shape, right_depth_video.shape)

            # item['left_depth_video'] = left_depth_video
            # item['right_depth_video'] = right_depth_video
            # print(left_depth_video.shape, right_depth_video.shape)

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f")"
        )

class VideoOnlyLazyDataset(LazyDatasetBase):
    def __init__(self, 
                 repo: dict|str = None,
                 root: Path | None = DATA_DIR,
                 split: str = "train",
                 ) -> None:
        super().__init__(repo, root, split)
    
    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset)

    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx) -> dict:
        '''Returns a dictionary containing the data for the given index.'''
        sample = deepcopy(self.hf_dataset[idx])

        videos_numpy = []
        for text, video_path in zip(sample["text"], sample["video"]):
            video_path = str(video_path) if isinstance(video_path, Path) else video_path

            if text["annotation_type"] == VideoDataBuilder.ANNOTATION_CONTINUOUS:
                st,ed = text["frame_indices"]
                frame_indices = list(range(st, ed))
            else:
                frame_indices = text["frame_indices"]
            video_clip = decode_video_torchvision(video_path, frame_indices, keyframes_only=True)
            videos_numpy.append(video_clip)

        sample['video'] = videos_numpy

        return sample

class ImageLazyDataset(LazyDatasetBase):
    def __init__(self, 
                 repo: dict|str = None,
                 root: Path | None = DATA_DIR,
                 split: str = "train",
                 ) -> None:
        super().__init__(repo, root, split)
    
    def __getitem__(self, idx) -> dict:
        '''Returns a dictionary containing the data for the given index.'''
        sample = deepcopy(self.hf_dataset[idx])

        images_numpy = []
        for img_path in sample["image"]:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # 确保是 3 通道
                    img_array = np.array(img, dtype=np.uint8)  # 转换为 numpy
                    images_numpy.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}", flush=True)
                images_numpy.append(None)  # 出错时填充 None
        
        sample['image'] = images_numpy

        return sample

    
class MultiVideoLazyDataset(torch.utils.data.Dataset):
    '''A dataset that combines multiple VideoLazyDatasets.'''
    def __init__(self, 
                 repos: list[dict],
                 root: Path | None = DATA_DIR,
                 split: list[str] | str = "train"
                 ) -> None:
        super().__init__()

        self.repo_ids = [repo['repo_id'] for repo in repos]
        splits = [split] * len(self.repo_ids) if isinstance(split, str) else split
        assert len(splits) == len(self.repo_ids), "Number of splits must match number of repositories."

        self._datasets = []
        for split_str, repo in zip(splits, repos):
            if repo['config']['class_type'] == 'image':
                self._datasets.append(
                    ImageLazyDataset(
                        repo,
                        root=root,
                        split=split_str
                    )
                )
            elif repo['config']['class_type'] == 'video':
                self._datasets.append(
                    VideoOnlyLazyDataset(
                        repo,
                        root=root,
                        split=split_str
                    )
                )
            else:
                self._datasets.append(
                    VideoLazyDataset(
                        repo,
                        root=root,
                        split=split_str
                    )
                )

        self.cumulative_episodes = [0]
        for dataset in self._datasets:
            self.cumulative_episodes.append(self.cumulative_episodes[-1] + dataset.num_episodes)

        self.root = root
    
    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.features.items()})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if key.startswith("observations"):
                keys.append(key)
        return keys

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)
    
    def __len__(self):
        return self.num_episodes
    
    @property
    def num_frames(self) -> int:
        """Total number of frames in the dataset."""
        return sum(d.num_frames for d in self._datasets)
    
    @property
    def frame_count_list(self) -> list[int]:
        """List of frame counts for each episode in the dataset."""
        return [d.frame_count_list for d in self._datasets]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        dataset_idx = bisect.bisect_right(self.cumulative_episodes, idx) - 1
        if dataset_idx < 0 or dataset_idx >= len(self._datasets):
            raise ValueError(f"Index {idx} is out of range for the datasets")
        start_idx = self.cumulative_episodes[dataset_idx]
        
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        item['dataset_name'] = self.repo_ids[dataset_idx]


        if 'default_instruction' not in item or len(item['default_instruction']) == 0:
            item['default_instruction'] = self._datasets[dataset_idx].config['default_instruction']
         
        item['trim_stationary'] = self._datasets[dataset_idx].config['trim_stationary']

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Root Path: '{self.root}',\n"
            f"  Number of Episodes: {self.num_episodes}"
            f")"
        )
import datasets
import re
from pathlib import Path
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_file

from dataclasses import dataclass, field
import pyarrow as pa
import warnings
from typing import Any, ClassVar, Dict, List, Tuple

from collections import defaultdict


import torch
import json

def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict

def join_dicts(data_dict_raw:dict, ep_dict_raw:dict, join_key:str='sample_name') -> dict:
    # 建立ep_dict的sample_name到索引的映射
    ep_name_to_idx = {name: idx for idx, name in enumerate(ep_dict_raw[join_key])}
    
    # 准备结果字典
    # result_dict = {col: [] for col in data_dict_raw.keys()}
    result_dict = defaultdict(list)
    # 添加ep_dict中独有的列
    for col in data_dict_raw.keys():
        if col not in result_dict:
            result_dict[col] = []
    
    # 遍历data_dict中的每一行
    for i, sample_name in enumerate(data_dict_raw[join_key]):
        # 添加data_dict中的数据
        for col in data_dict_raw.keys():
            result_dict[col].append(data_dict_raw[col][i])
        
        # 寻找对应的ep_dict数据
        if sample_name in ep_name_to_idx:
            ep_idx = ep_name_to_idx[sample_name]
            # 添加ep_dict中的数据（除了sample_name）
            for col in ep_dict_raw.keys():
                if col != join_key and col not in data_dict_raw.keys():
                    result_dict[col].append(ep_dict_raw[col][ep_idx])
        else:
            # 不存在对应数据，填充None
            for col in ep_dict_raw.keys():
                if col != join_key and col not in data_dict_raw.keys():
                    result_dict[col].append(None)
        
    return result_dict

@dataclass
class Annotation:
    """
    Provides a type for a dataset containing annotations.

    Example:

    ```python
    data_dict = [{"annotation_type": "single_frame", "annotation": ["jump", "run"], "frame_indices": [0, 1]}]
    features = {"annotation": Annotation()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct([
        ('annotation_type', pa.string()),
        ('annotation', pa.list_(pa.string())),
        ('frame_indices', pa.list_(pa.int32()))
    ])
    _type: str = field(default="Annotation", init=False, repr=False)

    def __call__(self):
        return self.pa_type

# @dataclass
# class VideoFrame:
#     # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
#     """
#     Provides a type for a dataset containing video frames.

#     Example:

#     ```python
#     data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
#     features = {"image": VideoFrame()}
#     Dataset.from_dict(data_dict, features=Features(features))
#     ```
#     """

#     # pa_type: ClassVar[Any] = pa.struct([{"path": pa.string(), "timestamp": pa.float32()}])
#     pa_type: ClassVar[Any] = pa.struct([
#         ('path', pa.string()),
#         ('timestamp', pa.float32())
#     ])
#     _type: str = field(default="VideoFrame", init=False, repr=False)

#     def __call__(self):
#         return self.pa_type

# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         "'register_feature' is experimental and might be subject to breaking changes in the future.",
#         category=UserWarning,
#     )
#     # to make VideoFrame available in HuggingFace `datasets`
#     register_feature(VideoFrame, "VideoFrame")

# split training and test data
def split_data(data, split):
    if split == "train":
        return data
    
    if "%" in split: # may be "train[INT%:]" or "train[:INT%]" or "train[:FLOAT%]" or "train[FLOAT%:]"
        match_from = re.search(r"train\[(\d+(?:\.\d+)?)%:\]", split)
        if match_from:
            from_frame_index = int(float(match_from.group(1))/100 * len(data))

        match_to = re.search(r"train\[:(\d+(?:\.\d+)?)%\]", split)
        if match_to:
            to_frame_index = int(float(match_to.group(1))/100 * len(data))
    else: # may be "train[INT:]" or "train[:INT]"
        match_from = re.search(r"train\[(\d+(?:\.\d+)?):\]", split)
        if match_from:
            from_frame_index = int(match_from.group(1))

        match_to = re.search(r"train\[:(\d+(?:\.\d+)?)\]", split)
        if match_to:
            to_frame_index = int(match_to.group(1))
        
    if match_from:
        data = data.select(range(from_frame_index, len(data)))
    elif match_to:
        data = data.select(range(to_frame_index))
    else:
        raise ValueError(
            f'`split` ({split}) should either be "train", "train[INT:]", "train[:INT]", \
                            "train[INT%:]", "train[:INT%]", "train[:FLOAT%]", or "train[FLOAT%:]"'
        )

    return data

def load_hf_dataset(repo_id: str,  root: Path, split: str) -> datasets.Dataset:
    """hf_dataset contains all the observations, states, actions, rewards, etc."""
    if root is not None:
        print(f"Loading dataset from {split}")
        hf_dataset = load_from_disk(str(Path(root) / repo_id / "train"))
        hf_dataset = split_data(hf_dataset, split)
    else:
        raise ValueError("root should not be None")

    return hf_dataset

def load_stats(repo_id, root) -> dict[str, dict[str, torch.Tensor]]:
    """stats contains the statistics per modality computed over the full dataset, such as max, min, mean, std

    Example:
    ```python
    normalized_action = (action - stats["action"]["mean"]) / stats["action"]["std"]
    ```
    """
    if root is  None:
        raise ValueError("root should not be None")
    path = Path(root) / repo_id / "meta_data" / "stats.safetensors"

    stats = load_file(path)
    return unflatten_dict(stats)


def load_info(repo_id, root) -> dict:
    """info contains useful information regarding the dataset that are not stored elsewhere

    Example:
    ```python
    print("frame per second used to collect the video", info["fps"])
    ```
    """
    if root is  None:
        raise ValueError("root should not be None")
    path = Path(root) / repo_id / "meta_data" / "info.json"

    with open(path) as f:
        info = json.load(f)
    return info

def load_config(repo_id, root) -> dict:
    """config contains the configuration used to collect the data

    Example:
        X2RDataProcessingConfig
    ```
    """
    if root is  None:
        raise ValueError("root should not be None")
    path = Path(root) / repo_id / "meta_data" / "data_config.json"

    with open(path) as f:
        config = json.load(f)
    return config

def load_videos(repo_id, root) -> Path:
    if root is not None:
        path = Path(root) / repo_id / "videos"
    else:
        raise ValueError("root should not be None")

    return path

def concatenate_episodes(ep_dicts):
    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict

def check_repo_id(repo_id: str) -> None:
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"""`repo_id` is expected to contain a community or user id `/` the name of the dataset
            (e.g. 'lerobot/pusht'), but contains '{repo_id}'."""
        )
    
def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> Dict[str, torch.Tensor]:
    """
    Calculate episode data index for the provided HuggingFace Dataset. Relies on episode_index column of hf_dataset.

    Parameters:
    - hf_dataset (datasets.Dataset): A HuggingFace dataset containing the episode index.

    Returns:
    - episode_data_index: A dictionary containing the data index for each episode. The dictionary has two keys:
        - "from": A tensor containing the starting index of each episode.
        - "to": A tensor containing the ending index of each episode.
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    The episode_index is a list of integers, each representing the episode index of the corresponding example.
    For instance, the following is a valid episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    Below, we iterate through the episode_index and populate the episode_data_index dictionary with the starting and
    ending index of each episode. For the episode_index above, the episode_data_index dictionary will look like this:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx
        else:
            # We are still in the same episode, so there is nothing for us to do here
            pass
    # We have reached the end of the dataset, so we append the ending location of the last episode to the "to" list
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index
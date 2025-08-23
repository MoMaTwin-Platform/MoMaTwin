from typing import Dict, List, Union
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

class TFData2TorchDataset(IterableDataset):

    def __init__(self, tf_dataset, length):
        # super().__init__(tf_dataset)
        super().__init__()
        self.tf_dataset = tf_dataset
        self._length = length

    def __iter__(self):
        return iter(self.tf_dataset)

    def __len__(self):
        return self._length

def get_tf_dataset(dataset_path, horizon=20, n_obs_steps=1, is_bi_mode=True):
    print(f"Loading Datasets : {dataset_path}")
    builder = tfds.builder_from_directory(dataset_path)
    dataset = builder.as_dataset()
    train_dataset, val_dataset = dataset['train'], dataset['validation']

    def _parse_fn(sample) -> tf.data.Dataset:
        left_actions = tf.concat([sample["actions"]["follow_left_ee_cartesian_pos"],
                    sample["actions"]["follow_left_ee_rotation"],
                    tf.reshape(sample["actions"]["follow_left_gripper"],(-1, 1))],
                    axis=1)
        right_actions = tf.concat([sample["actions"]["follow_right_ee_cartesian_pos"],
                    sample["actions"]["follow_right_ee_rotation"],
                    tf.reshape(sample["actions"]["follow_right_gripper"],(-1, 1))],
                    axis=1)
        if is_bi_mode:
            actions = tf.concat([left_actions, right_actions], axis=1)
        else:
            actions = right_actions # if single arm, right arm default
        # print(f'right_action shape: {right_actions.shape}')
        actions = actions[:horizon+n_obs_steps,:]
        # print(f'action shape: {actions.shape}')

        obs = sample['observations']
        obs = dict_apply(obs, lambda x : x[:n_obs_steps,...])
        return {
            'action': actions,
            'obs': obs,
        }
    train_dataset = train_dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset, val_dataset


def get_datasets(
    shape_meta: dict,
    dataset_path: List[str],
    horizon=1,
    n_obs_steps=None,
    n_latency_steps=0,
    seed=42,
    is_bi_mode=True):

    # get rgb and lowdim keys
    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    train_ds = []
    val_ds = []
    train_len = 0
    val_len = 0
    for dp in dataset_path:
        tds, vds = get_tf_dataset(dp, horizon=horizon, n_obs_steps=n_obs_steps, is_bi_mode=is_bi_mode)
        train_ds.append(tds)
        val_ds.append(vds)
        train_len += len(tds)
        val_len += len(vds)
    train_ds = tf.data.Dataset.sample_from_datasets(train_ds, None, stop_on_empty_dataset=False).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.sample_from_datasets(val_ds, None, stop_on_empty_dataset=False).prefetch(buffer_size=tf.data.AUTOTUNE)
    # convert torchdataset
    train_ds = TFData2TorchDataset(train_ds, train_len)
    val_ds = TFData2TorchDataset(val_ds, val_len)
    print(len(train_ds))
    print(len(val_ds))
    return train_ds,val_ds

def collate_wrapper(rgb_keys, n_obs_steps, action_dim):
    def framed_collate_fn(batch):
        actions= []
        agent_pos = []
        obs_dict = {}
        # print(batch)
        for rgb_key in rgb_keys:
            assert rgb_key in batch[0]['obs']
            if rgb_key not in obs_dict:
                obs_dict[rgb_key] = []
        # instructions,qpos = [],[]
        for data in batch:
            image_dict = data["obs"]
            # To,Nc,H,W,C
            image_dict = dict_apply(image_dict, lambda x : np.moveaxis(tfds.as_numpy(x), -1, 1))
            # image_dict = dict_apply(image_dict, lambda x : tf.transpose(x, perm=[0,3,1,2]))
            for image_key in obs_dict.keys():
                obs_dict[image_key].append(image_dict[image_key])
            # To,Nc,C,H,W
            # image = np.moveaxis(image, -1, 2)
            # print(f'image shape:{image.shape}')
            actions.append(tfds.as_numpy(data["action"][n_obs_steps:,:action_dim]))
            # sample as actions
            agent_pos.append(tfds.as_numpy(data["action"][:n_obs_steps, :action_dim]))
            # instructions.append(data["instructions"])
        actions = torch.as_tensor(np.array(actions))
        # print(images.shape)
        agent_pos = torch.as_tensor(np.array(agent_pos))
        # get obs_dict
        obs_dict = dict_apply(obs_dict, lambda x: torch.as_tensor(np.array(x)))
        obs_dict['agent_pos'] = agent_pos
        return {
                'action': actions,
                'obs': obs_dict
                # 'instruction': instructions,
        }
    return framed_collate_fn
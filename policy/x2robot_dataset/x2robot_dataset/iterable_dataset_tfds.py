import tensorflow as tf
import tensorflow_datasets as tfds

from typing import List, Union
from functools import partial 
import multiprocessing

import inspect
import glob

from collections import defaultdict
import numpy as np
import torch
import random



import threading
import queue
import multiprocessing

from x2robot_dataset.data_utils.dict_apply import dict_apply

from x2robot_dataset.common.data_preprocessing import (
    process_videos,
    process_action,
    process_instruction,
    _ACTION_KEY_EE_MAPPING_INV
)

def _wrap(f, is_flattened):
    """Wraps a method to return a X2RobotDataset instead of a tf.data.Dataset."""

    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if not isinstance(result, X2RobotDataset) and isinstance(result, tf.data.Dataset):
            # make the result a subclass of X2RobotDataset and the original class
            result.__class__ = type(
                "X2RobotDataset", (X2RobotDataset, type(result)), X2RobotDataset.__dict__.copy()
            )
            # propagate the is_flattened flag
            if is_flattened is None:
                result.is_flattened = f.__self__.is_flattened
            else:
                result.is_flattened = is_flattened
        return result

    return wrapper


class X2RobotDataset(tf.data.Dataset):
    def __getattribute__(self, name:str):
        # monkey-patches tf.data.Dataset methods to return X2Dataset
        attr = super().__getattribute__(name)
        if inspect.ismethod(attr):
            return _wrap(attr, None)
        return attr
    def _apply_options(self):
        """Applies some default options for performance."""
        options = tf.data.Options()
        options.autotune.enabled = True
        options.deterministic = False
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_fusion = True
        options.experimental_optimization.map_and_filter_fusion = True
        options.experimental_optimization.inject_prefetch = False
        options.experimental_warm_start = True
        return self.with_options(options)

    def with_ram_budget(self, gb: int):
        """Sets the RAM budget for the dataset. The default is half of the available memory.

        Args:
            gb (int): The RAM budget in GB.
        """
        options = tf.data.Options()
        options.autotune.ram_budget = gb * 1024 * 1024 * 1024  # GB --> Bytes
        return self.with_options(options)
    

    @staticmethod
    def make_dataset_from_rawdata(
        file_paths:List[str],
        obs_key:dict=None,
        preload_pool_size:int = 10,
        num_preloader_threads:int = 12,
    ) -> "X2RobotDataset":


        preload_queue =  queue.Queue(maxsize=preload_pool_size)
        file_list_queue = queue.Queue()

        def _preloader(file_paths):
            processed_count = 0
            total_count = len(file_paths)
            # try:
            while True:
                file_path = file_list_queue.get()
                if file_path is None:
                    processed_count += 1
                    if processed_count == total_count:
                        for _ in range(preload_pool_size):
                            preload_queue.put(None)
                    preload_queue.put(None)
                    break

                if not tf.io.gfile.isdir(file_path):
                    file_list_queue.task_done()
                    continue
                
                try:
                    video = process_videos(file_path, obs_key=obs_key)
                    actions = process_action(file_path, action_key_mapping = _ACTION_KEY_EE_MAPPING_INV) 

                    instructions = process_instruction(file_path)

                    preload_queue.put(
                        {
                            'observations':video, 
                            'actions':actions,
                            'instructions':instructions
                        }
                    )
                except:
                    print(f"Error processing file: {file_path}")
                    preload_queue.put(None)
            # finally:
            #     preload_queue.put(None)
            #     file_list_queue.task_done()

        def start_preloading_threads(file_paths):
            for _ in range(num_preloader_threads):
                threading.Thread(target=lambda:_preloader(file_paths)).start()


        def load_and_preprocess_data(file_paths):
            for file_path in file_paths:
                file_list_queue.put(file_path)
            for _ in range(num_preloader_threads):
                file_list_queue.put(None)

            # file_list_queue.join()

            # for _ in range(preload_pool_size):
            #     preload_queue.put(None)
        
        def dataset_generator(file_paths):
            none_count = 0
            
            while True:
                data_dict = preload_queue.get()
                if data_dict is None:
                    none_count += 1
                    if none_count == num_preloader_threads:
                        break
                    continue
                yield data_dict

        # def dataset_generator():
        #     while True:
        #         data_dict = preload_queue.get()
        #         if data_dict is None:
        #             break
        #         yield data_dict

        def create_dataset(file_paths):
            start_preloading_threads(file_paths)

            load_and_preprocess_data(file_paths)

            _signature = {
                'actions': 
                    {
                        'follow_right_ee_cartesian_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                        'follow_right_ee_rotation': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                        'follow_right_gripper': tf.TensorSpec(shape=[None,], dtype=tf.float32),
                        'follow_left_ee_cartesian_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
                        'follow_left_ee_rotation': tf.TensorSpec(shape=[None,3], dtype=tf.float32),
                        'follow_left_gripper': tf.TensorSpec(shape=[None], dtype=tf.float32),
                    },
                'observations':
                    {
                        'face_view':tf.TensorSpec(shape=[None, 480,640,3], dtype=tf.uint8), 
                        'left_wrist_view':tf.TensorSpec(shape=[None, 480,640,3], dtype=tf.uint8),
                        'right_wrist_view':tf.TensorSpec(shape=[None, 480,640,3], dtype=tf.uint8),
                    },
                'instructions':
                    {
                        # 'text_zh': tf.TensorSpec(shape=[None], dtype=tf.string),  # A list of text items
                        'text_en': tf.TensorSpec(shape=[None], dtype=tf.string),  # A list of text items
                        # 'seg_len': tf.TensorSpec(shape=[None,], dtype=tf.int32),
                    }
            }   

            dataset = _wrap(tf.data.Dataset.from_generator, False)(
                lambda: dataset_generator(file_paths),
                output_signature=_signature
            )._apply_options()

            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

            return dataset

        dataset = create_dataset(file_paths)
        return dataset, len(file_paths)


    @staticmethod
    def make_dataset_from_shard(
        dataset_path:str,
        split='train',
        num_parallel_calls:int=tf.data.AUTOTUNE,
        process_fn:callable=None
        ) -> "X2RobotDataset":

        shard_paths = glob.glob(f'{dataset_path}/shard_*') or [dataset_path]
        datasets = []
        num_data = 0
        for shard_path in shard_paths:
            builder = tfds.builder_from_directory(shard_path)
            if builder.info.splits[split].num_shards == 0:
                continue
            dataset = _wrap(builder.as_dataset, False)(
                split=split,
                shuffle_files=False,
                read_config=tfds.ReadConfig(
                    skip_prefetch=True,
                    num_parallel_calls_for_interleave_files=num_parallel_calls,
                    interleave_cycle_length=num_parallel_calls,
                ),
            )._apply_options() # shuffle_files=True
            if process_fn:
                dataset = process_fn(dataset)
            datasets.append(dataset)
            num_data += len(dataset)
       
        if not datasets:
            return None, 0

        mixed_ds = tf.data.Dataset.from_tensor_slices(datasets)
        mixed_ds = mixed_ds.interleave(
            lambda x: x, num_parallel_calls=num_parallel_calls,
            cycle_length=4,
            block_length=16
        )

        return mixed_ds, num_data
    
    @staticmethod
    def make_interleaved_dataset(
        dataset_paths:List[str],
        sample_ratio:List[float]=None,
        repeat_num:int=1,
        split:str='train',
        is_bi_mode=True,
        _parse_fn=None,
        from_rawdata=False,
        train_val_split=0.9,
        train_split_seed=42,
        preload_pool_size = 10,
        num_preloader_threads = 10,
        max_epoch=100,
        obs_keys=None
        ):

        datasets = []
        num_data = 0

        def _default_parse_fn(sample) -> tf.data.Dataset:
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

            merge_order = {'ceil_view':0, 'face_view':1, 'wall_view':2, 'left_wrist_view':3, 'right_wrist_view':4} 
            if "merged_view" in sample:
                images = sample['merged_view']
                wall_view = images[:,merge_order["wall_view"]]
                face_view = images[:,merge_order["face_view"]]
                left_wrist_view = images[:,merge_order["left_wrist_view"]]
                right_wrist_view = images[:,merge_order["right_wrist_view"]]
                observation = {
                    "wall_view": wall_view,
                    "face_view": face_view,
                    "left_wrist_view": left_wrist_view,
                    "right_wrist_view": right_wrist_view,
                }
            else:
                if obs_keys is not None:
                    observation = {}
                    for key in obs_keys:
                        observation[key] = sample['observations'][key]
                else:
                    observation = sample['observations']

            return {
                'action': actions,
                'observation': observation,
                "instruction": sample["instructions"]["text_en"][0]
            }

    
        def post_process_fn(dataset, parse_fn, ratio):
            dataset = dataset.with_ram_budget(8)
            dataset = dataset.take(int(ratio*len(dataset)))
            dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            return dataset
        
        
        _parse_fn = _parse_fn or _default_parse_fn
        if not from_rawdata:

            if sample_ratio is None:
                sample_ratio = [1.0] * len(dataset_paths)
            for dataset_path,ratio in zip(dataset_paths,sample_ratio):
                dataset, num = X2RobotDataset.make_dataset_from_shard(
                    dataset_path,
                    split=split,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    process_fn=partial(post_process_fn, parse_fn=_parse_fn, ratio=ratio)
                )
    
                if dataset and num > 0:
                    datasets.append(dataset)
                    num_data += num
        else:
            if sample_ratio is None:
                sample_ratio = [1.0] * len(dataset_paths)
            for dataset_path,ratio in zip(dataset_paths, sample_ratio):
                # file_paths = glob.glob(f"{dataset_path}/*")
                file_paths = [path for path in glob.glob(f"{dataset_path}/*") if path.split('/')[-1].startswith('sample')]

                random.seed(train_split_seed)
                random.shuffle(file_paths)

                train_val_split *= ratio
                train_len = int(train_val_split*len(file_paths))
                if split == 'train':
                    file_paths = file_paths[:train_len]  # 90% for training
                    file_paths *= max_epoch
                    random.shuffle(file_paths)
                elif split == 'validation':
                    file_paths = file_paths[train_len:]
                else:
                    raise ValueError(f"Invalid split: {split}")
                
                dataset, num = X2RobotDataset.make_dataset_from_rawdata(
                    file_paths,
                    preload_pool_size = preload_pool_size,
                    num_preloader_threads = num_preloader_threads,
                    obs_key=obs_keys
                )
                
                dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

                if dataset and num > 0:
                    datasets.append(dataset)
                    num_data += num

        if not datasets:
            return None, 0

        mixed_ds = tf.data.Dataset.from_tensor_slices(datasets)
        mixed_ds = mixed_ds.interleave(
            lambda x: x, num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=tf.data.AUTOTUNE,
            block_length=16
        ).repeat(repeat_num)
        num_data = num_data * repeat_num

        # mixed_ds = tf.data.experimental.sample_from_datasets(
        #     datasets, weights=None, stop_on_empty_dataset=False
        # )

        return mixed_ds, num_data
    
    
class ParallelIterator:
    def __init__(self,
                 tf_dataset: tf.data.Dataset,
                 total_len: int,
                 batch_size: int,
                 num_dp: int,
                 rank: int,
                 collate_fn=None):
        self.batch_size = batch_size
        self.macro_batch_size = batch_size * num_dp

        reminder = total_len % self.macro_batch_size
        self.data_length_per_process = total_len // self.macro_batch_size
        if reminder > 0:
            padding_samples = self.macro_batch_size - reminder
            tf_dataset = tf_dataset.concatenate(tf_dataset.take(padding_samples))

            self.data_length_per_process += 1


        self.tf_dataset = tf_dataset.batch(self.macro_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.rank = rank
        self._collate_fn = collate_fn

    def set_shuffling_seed(self,
                    epoch_id :int,
                    buffer_size:int = 8) -> None:
        self.tf_dataset = self.tf_dataset.shuffle(buffer_size=buffer_size, seed=epoch_id).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    def __iter__(self):
        for sample in self.tf_dataset:
            if self._collate_fn is not None:
                sample = self._collate_fn(sample)
            yield sample

    def __len__(self):
        return self.data_length_per_process
    

class FlatIterator:
    def __init__(self, 
                 tf_dataset : tf.data.Dataset,
                 max_buffersize : int =10000,
                 num_processes : int =12,
                 batch_size : int =1,
                 num_dp : int =1,
                 rank : int =0,
                 n_obs_steps : int =10,
                 horizon : int = 20, 
                 skip_every_n_frames : int = 1,
                 n_sub_samples : int = None,
                 n_preferences : int = 2,
                 seed: int = 42,
                 sampler_type: str = 'subsequence', # 'subsequence' or 'preference'
                 collate_fn : callable = None,
                 _sub_sampler : callable = None):
        self.tf_dataset = tf_dataset
        self.batch_size = batch_size
        self.macro_batch_size = batch_size * num_dp
        self.rank = rank
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.sampler_type = sampler_type
        self.skip_every_n_frames = skip_every_n_frames
        self._collate_fn = collate_fn

        np.random.seed(seed)
        self.queue = queue.Queue(maxsize=max_buffersize)
        self.threads = []
        self.stop_event = threading.Event()

        if _sub_sampler is not None:
            self._sub_sampler = _sub_sampler
        else:
            self._sub_sampler = SubSampler(self.queue, n_obs_steps, horizon, skip_every_n_frames,
                                           n_samples=n_sub_samples,
                                           n_preferences=n_preferences)
            self._sub_sampler = partial(self._sub_sampler, sampler_type=self.sampler_type)
        
    
        self.start_threads(num_threads=num_processes)

            
    def start_threads(self, num_threads):
        for _ in range(num_threads):
            thread = threading.Thread(target=self._producer)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def _producer(self):
        while not self.stop_event.is_set():
            for sample in self.tf_dataset:
                self._sub_sampler(sample)
    
    def stop_threads(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
    '''
    # use tf.data.Dataset
    def __iter__(self):
        # dataset = dataset.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
        # dataset = dataset.map(lambda *x: reconstruct_dict(*x))
        # dataset = dataset.batch(self.macro_batch_size, drop_remainder=False)
        dataset = self.tf_dataset.map(self._sub_sampler, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
        dataset = dataset.batch(self.macro_batch_size, drop_remainder=False)

        for batch in dataset:
            batch_dict = {
                'observation': {key: batch['observation'][key].numpy() for key in batch['observation'].keys()},
                'action': batch['action'].numpy(),
                'instruction': batch['instruction'].numpy()
            }

            if self._collate_fn is not None:
                yield self._collate_fn(batch_dict)
            else:
                yield batch_dict
    
    # single thread version
    def __iter__(self):
        while True:
            buffer,batch = [],[]
            for sample in self.tf_dataset:
                subsequence = self._sub_sampler(sample)
                buffer.extend(subsequence)
                if len(buffer) >= self.macro_batch_size:
                    batch = buffer[:self.macro_batch_size]
                    buffer = buffer[self.macro_batch_size:]
                    break
            
            if not batch:
                return
            
            batch = {
                'observation': {key: np.stack([subsequence['observation'][key] for subsequence in batch], axis=0)
                                for key in batch[0]['observation'].keys()},
                'action': np.stack([subsequence['action'] for subsequence in batch], axis=0),
                'instruction': np.array([subsequence['instruction'] for subsequence in batch])
            }

            if self._collate_fn is not None:
                yield self._collate_fn(batch)
            else:
                yield batch
    '''

    def __iter__(self):
        while True:
            batch = []
            try:
                # tic = time.time()
                for _ in range(self.macro_batch_size):
                    subsequence = self.queue.get(timeout=1800)  # Added timeout to avoid deadlock
                    batch.append(subsequence)

            except queue.Empty:
                if self.stop_event.is_set() and self.queue.empty():
                    self.stop_threads()
                    if batch:
                        break
                    else:
                        return
            
            if self.sampler_type == 'subsequence':
                batch = {
                    'observation': {key: np.stack([subsequence['observation'][key] for subsequence in batch], axis=0)
                                    for key in batch[0]['observation'].keys()},
                    'action': np.stack([subsequence['action'] for subsequence in batch], axis=0),
                    'instruction': np.array([subsequence['instruction'] for subsequence in batch])
                }
            elif self.sampler_type == 'preference':
                batch = {
                    'observation': {key: np.stack([subsequence['observation'][key] for subsequence in batch], axis=0)
                                    for key in batch[0]['observation'].keys()},
                    'instruction': np.array([subsequence['instruction'] for subsequence in batch]),
                    'indices': np.stack([subsequence['indices'] for subsequence in batch], axis=0)
                }
            else:
                raise ValueError(f'Invalid sampler type: {self.sampler_type}')

            if self._collate_fn is not None:
                yield self._collate_fn(batch)
            else:
                yield batch

    def __len__(self):
        return 1000000  # Dummy value, as we do not know the length of the dataset

class SubSampler:
    def __init__(self, queue, n_obs_steps, horizon, skip_every_n_frames=1, n_samples=None, n_preferences=2):
        self.queue = queue
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon 
        self.skip_every_n_frames = skip_every_n_frames
        self.n_samples = n_samples
        self.n_preferences = n_preferences

    def _pad_seq(self, seq, left_pad, right_pad):
        if left_pad > 0:
            left_padding = np.full((left_pad, *seq.shape[1:]), seq[0:1].copy(), dtype=seq.dtype)
            seq = np.concatenate([left_padding, seq])
        if right_pad > 0:
            right_padding = np.full((right_pad, *seq.shape[1:]), seq[-1:].copy(), dtype=seq.dtype)
            seq = np.concatenate([seq, right_padding])
        return seq

    def _sample_subsequences(self, sample):
        length = sample['observation']['face_view'].shape[0]
        
        padded_observations = {
            key: self._pad_seq(val.numpy(), self.n_obs_steps*self.skip_every_n_frames, 0) 
            for key, val in sample['observation'].items()
        }
        
        padded_actions = self._pad_seq(sample['action'].numpy(), self.n_obs_steps*self.skip_every_n_frames, self.horizon)

        size = self.n_samples if self.n_samples is not None else length
        start_indices = np.random.randint(self.n_obs_steps * self.skip_every_n_frames, 
                                          self.n_obs_steps * self.skip_every_n_frames + length, 
                                          size=size)
        
        for obs_slice_end in start_indices:
            obs_slice_start = obs_slice_end - self.n_obs_steps * self.skip_every_n_frames
            
            act_slice_skipped_start = obs_slice_start
            act_slice_skipped_end = obs_slice_end
            act_slice_continuous_start = obs_slice_end
            act_slice_continuous_end = act_slice_continuous_start + self.horizon
            
            if act_slice_continuous_end > len(padded_actions):
                continue
                
            actions_skipped = padded_actions[act_slice_skipped_start:act_slice_skipped_end:self.skip_every_n_frames]
            actions_continuous = padded_actions[act_slice_continuous_start:act_slice_continuous_end]
            
            combined_actions = np.concatenate((actions_skipped, actions_continuous), axis=0)
            
            self.queue.put({
                'action': combined_actions,
                'observation': {
                    key: val[obs_slice_start:obs_slice_end:self.skip_every_n_frames]
                    for key, val in padded_observations.items()  
                },
                'instruction': sample['instruction']
            })
        
    def _sample_preference(self, sample):
        # print('sample', sample)
        length = sample['observation']['face_view'].shape[0]
        
        frame_indices = np.arange(length)

        n_samples = self.n_samples if self.n_samples is not None else length
        
        for _ in range(n_samples):
            preference_indices = np.random.choice(frame_indices, size=self.n_preferences, replace=False)
            preference_indices.sort()
            
            preference_observations = {
                key: tf.gather(val, preference_indices) for key, val in sample['observation'].items()
            }
        
            self.queue.put({
                'observation': preference_observations,
                'instruction': sample['instruction'],
                'indices': preference_indices
            })
                    
    def __call__(self, sample, sampler_type:str = 'subsequence'):
        if sampler_type == 'subsequence':
            self._sample_subsequences(sample)
        elif sampler_type == 'preference':
            self._sample_preference(sample)
        else:
            raise ValueError(f'Invalid sampler type: {sampler_type}')

def collate_wrapper(rgb_keys, n_obs_steps, horizon, action_dim, rank, batch_size):
    def collate_fn(macro_batch):
        actions= []
        agent_pos = []
        obs_dict = {}

        for rgb_key in rgb_keys:
            assert rgb_key in macro_batch['observation']

        rank_start,rank_end = rank*batch_size, (rank+1)*batch_size
        image_dict = {k:macro_batch['observation'][k] for k in rgb_keys}

        if isinstance(macro_batch['action'], tf.Tensor):
            image_dict = dict_apply(image_dict, lambda x : np.moveaxis(tfds.as_numpy(x), -1, 2))

            obs_dict = dict_apply(image_dict, lambda x : x[rank_start:rank_end, \
                                                        :n_obs_steps, ...])
            obs_dict = dict_apply(obs_dict, lambda x: torch.as_tensor(np.array(x)))
            
            actions = tfds.as_numpy(macro_batch["action"][rank_start:rank_end, \
                                                        n_obs_steps:n_obs_steps+horizon, \
                                                        :action_dim])
            actions = torch.as_tensor(actions)
            agent_pos = tfds.as_numpy(macro_batch["action"][rank_start:rank_end, \
                                                        :n_obs_steps, \
                                                        :action_dim])
            obs_dict['agent_pos'] = torch.as_tensor(agent_pos)
        else:
            image_dict = dict_apply(image_dict, lambda x : np.moveaxis(x, -1, 2))

            obs_dict = dict_apply(image_dict, lambda x : x[rank_start:rank_end, \
                                                        :n_obs_steps, ...])
            obs_dict = dict_apply(obs_dict, lambda x: torch.as_tensor(np.array(x)))
            
            actions = macro_batch["action"][rank_start:rank_end, \
                                            n_obs_steps:n_obs_steps+horizon, \
                                            :action_dim]
            actions = torch.as_tensor(actions)
            agent_pos = macro_batch["action"][rank_start:rank_end, \
                                            :n_obs_steps, \
                                            :action_dim]
            obs_dict['agent_pos'] = torch.as_tensor(agent_pos)


        instructions = macro_batch['instruction'][rank_start:rank_end]
        return {
                'action': actions,
                'obs': obs_dict,
                'instruction': instructions
        }
    return collate_fn

def prefrenece_collate_wrapper(rgb_keys, rank, batch_size):
    def collate_fn(macro_batch):
        for rgb_key in rgb_keys:
            assert rgb_key in macro_batch['observation']

        rank_start,rank_end = rank*batch_size, (rank+1)*batch_size
        image_dict = {k:macro_batch['observation'][k] for k in rgb_keys}

        if isinstance(next(iter(image_dict.values())), tf.Tensor):
            image_dict = dict_apply(image_dict, lambda x : np.moveaxis(tfds.as_numpy(x), -1, 2))

            image_dict = dict_apply(image_dict, lambda x : x[rank_start:rank_end, ...])
            image_dict = dict_apply(image_dict, lambda x: torch.as_tensor(np.array(x)))            
        else:
            image_dict = dict_apply(image_dict, lambda x : np.moveaxis(x, -1, 2))

            image_dict = dict_apply(image_dict, lambda x : x[rank_start:rank_end, ...])
            image_dict = dict_apply(image_dict, lambda x: torch.as_tensor(np.array(x)))

        instructions = macro_batch['instruction'][rank_start:rank_end]
        indices = macro_batch['indices'][rank_start:rank_end]
    
        return {
                'obs': image_dict,
                'instruction': instructions,
                'indices': indices
        }
    return collate_fn
 

import glob
import random
if __name__ == '__main__':
    dataset_path = '/x2robot/zhengwei/10000/20240501-clean-dish-addition'
    file_paths = [path for path in glob.glob(f"{dataset_path}/*") if path.split('/')[-1].startswith('sample')]
    random.seed(42)
    random.shuffle(file_paths)

    dataset, num_data = X2RobotDataset.make_interleaved_dataset(
        dataset_paths =[dataset_path],
        split='train',
        is_bi_mode=True,
        from_rawdata=True,
        train_val_split=0.9,
        train_split_seed=42,
        preload_pool_size = 10,
        num_preloader_threads = 16,
        max_epoch=1000)
    

    # dataset, count = X2RobotDataset.make_dataset_from_rawdata(file_paths=file_paths,
    #                                                           preload_pool_size = 10,
    #                                                           num_preloader_threads = 16)

    # is_bi_mode = True
    # obs_keys = None
    # def _parse_fn(sample) -> tf.data.Dataset:
    #     left_actions = tf.concat([sample["actions"]["follow_left_ee_cartesian_pos"],
    #                 sample["actions"]["follow_left_ee_rotation"],
    #                 tf.reshape(sample["actions"]["follow_left_gripper"],(-1, 1))],
    #                 axis=1)
    #     right_actions = tf.concat([sample["actions"]["follow_right_ee_cartesian_pos"],
    #                 sample["actions"]["follow_right_ee_rotation"],
    #                 tf.reshape(sample["actions"]["follow_right_gripper"],(-1, 1))],
    #                 axis=1)
    #     if is_bi_mode:
    #         actions = tf.concat([left_actions, right_actions], axis=1)
    #     else:
    #         actions = right_actions # if single arm, right arm default

    #     merge_order = {'ceil_view':0, 'face_view':1, 'wall_view':2, 'left_wrist_view':3, 'right_wrist_view':4} 
    #     if "merged_view" in sample:
    #         images = sample['merged_view']
    #         wall_view = images[:,merge_order["wall_view"]]
    #         face_view = images[:,merge_order["face_view"]]
    #         left_wrist_view = images[:,merge_order["left_wrist_view"]]
    #         right_wrist_view = images[:,merge_order["right_wrist_view"]]
    #         observation = {
    #             "wall_view": wall_view,
    #             "face_view": face_view,
    #             "left_wrist_view": left_wrist_view,
    #             "right_wrist_view": right_wrist_view,
    #         }
    #     else:
    #         if obs_keys is not None:
    #             observation = {}
    #             for key in obs_keys:
    #                 observation[key] = sample['observations'][key]
    #         else:
    #             observation = sample['observations']

    #     return {
    #         'action': actions,
    #         'observation': observation,
    #         "instruction": sample["instructions"]["text_en"][0]
    #     }

    # dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # import tqdm
    # for data in tqdm.tqdm(dataset):
    #     print(data.keys())
    collate_fn = prefrenece_collate_wrapper(rgb_keys=["face_view", "left_wrist_view", "right_wrist_view"], 
                                            rank=0,
                                            batch_size=128)
    # collate_fn = collate_wrapper(rgb_keys=["face_view", "left_wrist_view", "right_wrist_view"],
    #                             n_obs_steps = 10, horizon = 20, action_dim=14, rank=1, batch_size=32)

    iterator = FlatIterator( 
                 tf_dataset = dataset,
                 max_buffersize = 10000,
                 num_processes = 8,
                 batch_size = 128,
                 num_dp  = 1,
                 rank =0,
                 n_obs_steps = 10,
                 horizon  = 20,
                 n_sub_samples = 1024,
                 n_preferences=6,
                 skip_every_n_frames = 1,
                 collate_fn=collate_fn,
                 sampler_type='preference',
                # sampler_type='subsequence',
                 seed = 42)
    
    import tqdm
    for i, batch in enumerate(tqdm.tqdm(iterator)):
        pass

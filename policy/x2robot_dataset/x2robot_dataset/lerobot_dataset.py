from typing import Dict, List
from pathlib import Path

from x2robot_dataset.ler_common.prepare_data import LeRDataBuilder
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
import tqdm
from dataclasses import dataclass, field

import torch
import glob
import os

from x2robot_dataset.common.data_preprocessing import _TAC_FILE_MAPPING

@dataclass
class LeRDataConfig:
    cam_mapping:dict            = field(default_factory=dict)
    tac_mapping:Dict[str,str]   = field(default_factory=dict)
    tac_dim:tuple               = (3,15)
    head_action_dim:tuple       = (2,)
    image_height:int            = 480
    image_width:int             = 640
    fps:int                     = 20
    filter_angle_outliers:bool  = True
    detect_motion:bool          = True
    trim_stationary:bool        = True
    parse_tactile:bool          = False
    parse_head_action:bool      = False
    train_test_split:float      = 0.9
    split_seed:int              = 42

    def __post_init__(self):
        self.cam_mapping = {
            'faceImg':'face_view',
            'leftImg':'left_wrist_view',
            'rightImg':'right_wrist_view'
        }
        self.tac_mapping = _TAC_FILE_MAPPING

    def as_dict(self):
        return self.__dict__



def make_lerobot_dataset(*,
    data_folders:List[str],
    data_configs:List[Dict],
    rank:int = 0,
    force_overwrite:bool = False,
    split:str = 'train',
    root_dir = Path('/x2robot/Data/.cache/hf_datasets')
):
    assert len(data_folders) == len(data_configs), "Data folders and data configs should have the same length"

    data_groups = []
    for dataset_path, single_config in zip(data_folders, data_configs):
        single_folder = []
        for entry in glob.glob(f'{dataset_path}/*'):
            if os.path.isdir(entry):
                has_mp4_files = len(glob.glob(f'{entry}/*.mp4')) > 0
                if has_mp4_files:
                    single_folder.append(entry)
        data_groups.append((single_folder, single_config))
    
    repo_ids, splits = [], []
    if rank == 0:
        for (video_files, data_config) in tqdm.tqdm(data_groups, desc="Processing data folders"):
            out_repo_id,_ = LeRDataBuilder(data_config).from_raw_to_lerobot_format(
                                            video_files=video_files,
                                            force_overwrite=force_overwrite,
                                            root_dir=root_dir,
                                        )
            repo_ids.append(out_repo_id)
            percentage = data_config['train_test_split']*100
            split_expr = f'train[:{percentage}%]' if split == 'train' else f'train[{percentage}%:]'
            splits.append(split_expr)

    # if dist is initialized, wait for rank 0 to finish
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

            
    dataset = MultiLeRobotDataset(
            repo_ids=repo_ids,
            root=root_dir,
            split=splits,
    )

    return dataset

class X2DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self,
                 action_sampling_rate=1.0,                
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
            
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        '''
        Get a chunk from the dataset
        '''
        try:
            idx = self.available_indices[index][0]
            eposide_end = self.available_indices[index][1]

            actions = {}
            action_histories = {}
            
            observations = {}
            observation_histories = {}
            observation_goal = {}
            
            action_st_idx, action_ed_idx = self.action_start_point+idx, self.action_start_point+idx+self.action_horizon
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

                action_history_start = max(action_st_idx - self.action_history_length, 0)
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

                history_start = max(obs_st_idx - self.image_history_length, 0)
                obs_history = np.array(self.replay_buffer.data[key][history_start:obs_st_idx])
                if self.left_padding:
                    pad_size = self.image_history_length - obs_history.shape[0]
                    obs_history = self._pad_with_value(obs_history, pad_size, padding_dir='left')
                observation_histories[key] = obs_history

                obs_goal = np.array(self.replay_buffer.data[key][obs_goal_idx])
                observation_goal[key] = obs_goal
            
            frame_idx,uid = self._find_episode_name(idx)
            if self._get_instruction:
                instructions = self._get_instruction(uid=uid, frames=range(frame_idx,frame_idx+self.action_horizon))
                instruction_histories = self._get_instruction(uid=uid, frames=range(frame_idx-self.action_history_length, frame_idx))
    
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
            valid_indices = [(idx, end) for start, end in zip(self.starts, self.ends) \
                            for idx in range(start, end, self.action_sampling_step)]

            return valid_indices

        valid_indices = [(idx,end) for start, end in zip(self.starts, self.ends) \
                        for idx in range(start, end, self.action_sampling_step) \
                            if idx + self.action_horizon < end]
        

        return valid_indices
    
    def _find_episode_name(self, frame_idx):
        for name, start, end in zip(*[self.sample_names, self.starts, self.ends]):
            if start <= frame_idx < end:
                return frame_idx-start,name

        return None,None


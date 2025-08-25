import numpy as np
import random
from collections import defaultdict
import torch
import pickle
import time
import functools
import copy
from typing import List
from einops import rearrange
from x2robot_dataset.dict_codec import TensorCodec
import torchvision.transforms as TT


from x2robot_dataset.common.lie_group import (
    so3_to_lie_algebra,
    euler_to_rotation_matrix
)
from x2robot_dataset.common.mask import (
    overlay_masks,
    overlay_mask_on_image,
    overlay_box_on_image
)
dof_config = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation": 3,
        "follow_left_gripper": 1,
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation": 3,
        "follow_right_gripper": 1,
        "head_actions": 3,
        "car_pose": 3,
        "height": 1,
        "base_velocity": 2,
    }
'''
follow_left_arm_joint_pos, follow_left_arm_joint_dev, follow_left_arm_joint_cur
follow_right_arm_joint_pos, follow_right_arm_joint_dev, follow_right_arm_joint_cur
'''

dof_config_joint = {
    "follow_left_arm_joint_pos": 7,
    "follow_right_arm_joint_pos": 7,
}

from x2robot_dataset.common.data_utils import actions_to_relative

class CollateHelper:
    @staticmethod
    def stack_action(actions, 
                     parse_head_action=False, 
                     fill_zero=False, 
                     to_lie_algebra=False, 
                     # action_keys: 需要stack的action的key, 默认是双臂动作
                     action_keys=['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper'], 
                     relative_action=False, 
                     add_noise=False, 
                     use_joint_action=False,
                     add_origin=False,
                     ):
        def _convert_to_lie_algebra(rotations):
            ws = []
            for euler_angles in rotations:
                w = so3_to_lie_algebra(euler_to_rotation_matrix(euler_angles))
                ws.append(w)
            return np.stack(ws, axis=0, dtype=rotations[0].dtype)
        
        convert_to_lie_algebra = _convert_to_lie_algebra if to_lie_algebra else lambda x: x
        # car_pose -> base_velocity
        if "car_pose" in actions:
            car_pose = actions["car_pose"]  # shape: (n, 3)
            dt = 1/20  # 采样频率20Hz

            # 计算相邻帧之间的差值
            pose_diff = car_pose[1:] - car_pose[:-1]  # shape: (n-1, 3)        
            linear_velocity = np.linalg.norm(pose_diff[:, :2], axis=1) / dt  # shape: (n-1,)
            angular_velocity = pose_diff[:, 2] / dt  # shape: (n-1,)
            base_velocity = np.stack([linear_velocity, angular_velocity], axis=1)  # shape: (n-1, 2)
            # 在开头添加零速度 as agent_pos
            base_velocity = np.vstack([np.zeros((1, 2)), base_velocity])  # shape: (n, 2)
            actions["base_velocity"] = base_velocity
            actions.pop("car_pose")

        is_bi_mode = "follow_left_ee_cartesian_pos" in actions or "follow_left_arm_joint_pos" in actions

        for key, val in actions.items():
            if key.startswith('follow_right') and len(val) == 0:
                return np.array([])
        
        all_actions = []
        action_chunk_size = actions[list(actions.keys())[0]].shape[0]
        if action_keys is not None:
            for key in action_keys:
                if key=="follow_left_ee_cartesian_pos" and add_origin:
                    actions[key] += np.array([0.105, 0, 0.055])
                if key=="follow_right_ee_cartesian_pos" and add_origin:
                    actions[key] += np.array([0.105, 0, 0.055])
                if key not in actions:
                    print(f"key: {key} not in actions")
                    actions[key] = np.zeros((action_chunk_size,dof_config[key]))
                if len(actions[key].shape) == 1:
                    all_actions.append(actions[key].reshape(-1, 1))
                else:
                    all_actions.append(actions[key])
            
            all_actions = np.concatenate(all_actions, axis=1)
            if relative_action:
                # TODO 没有验证，需要改actions_to_relative
                all_actions = actions_to_relative(all_actions, add_noise=add_noise)
            
            return all_actions
        else:
            raise NotImplementedError("In collate_fn/stack_action, action_keys is None!!!")
        
        # right_actions = np.concatenate([
        #         actions["follow_right_ee_cartesian_pos"],
        #         convert_to_lie_algebra(actions["follow_right_ee_rotation"]),
        #         actions["follow_right_gripper"].reshape(-1, 1)], axis=1
        # ) if not use_joint_action else actions["follow_right_arm_joint_pos"]

        # if add_origin:
        #     # 如果是工厂数据，需要对右臂的动作进行一些微调，x轴+0.15，z轴+0.05
        #     right_actions[:, 0] += 0.10
        #     right_actions[:, 2] += 0.05

        # if not use_joint_action:
        #     left_actions = np.concatenate([
        #             actions["follow_left_ee_cartesian_pos"],
        #             convert_to_lie_algebra(actions["follow_left_ee_rotation"]),
        #             actions["follow_left_gripper"].reshape(-1, 1)], axis=1
        #         ) if is_bi_mode else np.zeros_like(right_actions)
        # else:
        #     left_actions = actions["follow_left_arm_joint_pos"] if is_bi_mode else np.zeros_like(right_actions)

        # if add_origin:
        #     # 如果是工厂数据，需要对右臂的动作进行一些微调，x轴+0.15，z轴+0.05
        #     left_actions[:, 0] += 0.10
        #     left_actions[:, 2] += 0.05

        # stacked_actions = np.concatenate([left_actions, right_actions], axis=1) if (is_bi_mode or fill_zero) else right_actions

        # if relative_action:
        #     stacked_actions = actions_to_relative(stacked_actions, add_noise=add_noise) # 按照第一帧进行相对位置计算，第一帧后面会作为agent_pos

        # if parse_head_action and "head_actions" in actions:
        #     head_actions = actions['head_actions']
        #     stacked_actions = np.concatenate([stacked_actions, head_actions], axis=1)
        # return stacked_actions
    
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
        if new_instruction is None:
            new_instruction = ""
        subtask_generation = sample.get("subtask_generation", None)
        if subtask_generation is not None:
            if random.random() < 0.3:
                new_instruction += subtask_generation
            else:
                new_instruction = subtask_generation
        return new_instruction
    

def full_collate_fn(batch, obs_keys, action_dim, is_bi_mode, parse_head_action=False, to_lie_algebra=False, action_keys=None):
    batch = list(filter(lambda x: x is not None, batch))
    batched_dict = defaultdict(list)

    for sample in batch:
        if "car_pose" in sample["actions"] and sample["actions"]["car_pose"][0,0]>10:
            print(f"sample: {sample['uid']} car_pose: {sample['actions']['car_pose']}",flush=True)
            continue
        actions = CollateHelper.stack_action(
                sample["actions"],
                parse_head_action=parse_head_action,
                to_lie_algebra=to_lie_algebra,
                action_keys=action_keys,
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


def chunk_collate_fn(batch,
                    #  obs_keys,
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
                     relative_action=False, # 是否是相对动作
                     add_noise=False, # 是否添加噪声
                     action_keys=None,
                     agent_pos_keys=None,
                     use_joint_action=False,
                     use_gripper_cur=False, # 是否使用gripper的力矩
                     use_joint_cur=False, # 是否使用所有关节的力矩
                     ):
    # mask_type: mask_only return mask, mask_on_image return image(face_view) with mask

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    start_time = time.time()

    # for mask
    batched_mask_dict = {mask_key:[] for mask_key in mask_keys} if mask_keys else {}
    batched_dict = defaultdict(list)

    def _pad_zero(seq:List[torch.Tensor], max_len=None) -> torch.Tensor:
        max_len = max([len(item) for item in seq]) if max_len is None else max_len
        for i in range(len(seq)):
            seq[i] = torch.cat([seq[i], torch.zeros(max_len - len(seq[i]), dtype=torch.int32)])
        
        return torch.stack(seq)


    for sample in batch:
        # if sample2imginstruct is not None and sample['uid'] not in sample2imginstruct:
        #     continue
        add_origin = False

        robot_id = sample["uid"].split("_")[0]
        if robot_id in ["factory10002", "factory10003", "factory10004", "factory10005", 
                        "factory10006", "factory10007", "factory10008", "factory10009", 
                        "factory10010", "factory10011", "factory10012", "factory10013",
                        "factory10014",]:
            # print(sample["uid"], "add origin")
            add_origin = True

        # sample_actions_copy = copy.deepcopy(sample["actions"])
        actions = CollateHelper.stack_action(
                    sample["actions"], 
                    parse_head_action=parse_head_action,
                    fill_zero=True,
                    to_lie_algebra=to_lie_algebra,
                    relative_action=relative_action,
                    add_noise=add_noise,
                    action_keys=action_keys,
                    use_joint_action=use_joint_action,
                    add_origin=add_origin,
        ) # (action_horizon+low_dim_obs_horizon, action_dim) ## +low_dim_obs_horizon(一般是1) 因为第一帧是agent_pos
        batched_dict['action'].append(actions)

        if agent_pos_keys is None:
            agent_pos = actions[:low_dim_obs_horizon, :]
        else:
            agent_pos = CollateHelper.stack_action(
                sample["actions"], 
                parse_head_action=parse_head_action,
                fill_zero=True,
                to_lie_algebra=to_lie_algebra,
                action_keys=agent_pos_keys,
            )
            agent_pos = agent_pos[:low_dim_obs_horizon, :] # (low_dim_obs_horizon, action_dim)

        batched_dict['agent_pos'].append(agent_pos)

        action_history = CollateHelper.stack_action(
                sample["action_histories"],
                parse_head_action=parse_head_action,
                fill_zero=True,
                to_lie_algebra=to_lie_algebra,
                relative_action=relative_action,
                add_noise=add_noise,
                action_keys=action_keys,
                use_joint_action=use_joint_action,
                add_origin=add_origin,
            ) # (action_history_length, action_dim)

        batched_dict['action_history'].append(action_history)

        obs_keys = list(sample["observations"].keys()) # it only contains camera images
        observation = CollateHelper.stack_obs(sample["observations"], obs_keys)
        observation_goal = CollateHelper.stack_obs_goal(sample["observation_goal"], obs_keys)
        observation_history = {}
        if merge_image_history:
            observation_history = CollateHelper.stack_obs(sample["observation_histories"], obs_keys)
            batched_dict['obs_history'].append(observation_history)
        if 'next_observation' in sample:
            next_observation = CollateHelper.stack_obs(sample["next_observation"], obs_keys)
            batched_dict['next_obs'].append(next_observation)
        batched_dict['obs'].append(observation)
        batched_dict['obs_goal'].append(observation_goal)

        new_instruction = CollateHelper.stack_instruction(sample, sample2instruct)
        if new_instruction is None:
            # print(f"new_instruction is None: {sample['uid']}",flush=True)
            new_instruction = ""
        batched_dict['instruction'].append(new_instruction)
    
        if mask_type:
            assert mask_type in ['mask_only', 'mask_on_image', 'box_on_image'], f'{mask_type} is not support!'
            for mask_key in mask_keys:
                mask = sample[mask_key]
                if mask_type == 'mask_on_image': # not used now
                    mask = overlay_mask_on_image(image=observation['face_view'],  mask=mask)
                elif mask_type == 'box_on_image': # not used now
                    mask = overlay_box_on_image(image=observation['face_view'],  box=mask)
                elif mask_type == 'mask_only':
                    mask = overlay_masks(mask=mask, T=img_obs_horizon)
                batched_mask_dict[mask_key].append(mask)

        batched_dict['uid'].append(sample['uid'])
        batched_dict['frame'].append(sample['frame'])
        batched_dict['tactile'].append(sample['tactiles']) if 'tactiles' in sample else None
        batched_dict['joint_cur'].append(sample['joint_cur']) if 'joint_cur' in sample else None
        batched_dict['joint_cur_history'].append(sample['joint_cur_history']) if 'joint_cur_history' in sample else None

    actions = np.stack(batched_dict['action'])
    agent_pos = np.stack(batched_dict['agent_pos'])
    # rearrange obs from (bs, img_obs_horizon, channel, height, width) to (bs, img_obs_horizon, height, width, channel)
    # first check if the last dimension is channel
    
    obs_dict = {
        key : torch.as_tensor(
                np.stack([item[key] for item in batched_dict['obs']])[:, :img_obs_horizon, ...]
            ) for key in obs_keys
    }
    if merge_image_history:
        obs_history = {
            key : torch.as_tensor(
                np.stack([item[key] for item in batched_dict['obs_history']])
            ) for key in obs_keys
        }
        for key in obs_keys:
            obs_dict[key] = torch.concat([obs_dict[key], obs_history[key]], dim=1)
    obs_dict_goal = torch.as_tensor(np.array(batched_dict['obs_goal']))
    # add next_obs_dict
    next_obs_dict = {
        key : torch.as_tensor(
                np.stack([item[key] for item in batched_dict['next_obs']])
            ) for key in obs_keys
    } if 'next_obs' in batched_dict else None
    first_key = obs_keys[0]
    if obs_dict[first_key].shape[-1] != 3:
        obs_dict = {
            key: rearrange(obs_dict[key], 'b t c h w -> b t h w c') for key in obs_keys
        }
        obs_dict_goal = rearrange(obs_dict_goal, 'b c h w -> b h w c')
    
    # obs_dict['agent_pos'] = torch.as_tensor(actions[:, :low_dim_obs_horizon, :action_dim]) # (bs, low_dim_obs_horizon, action_dim)
    obs_dict['agent_pos'] = torch.as_tensor(agent_pos[:, :low_dim_obs_horizon, :action_dim])

    if use_gripper_cur:
        left_gripper_cur =  torch.tensor(np.stack([cur["follow_left_arm_joint_cur"][:,-1] for cur in batched_dict['joint_cur']])).unsqueeze(1) #[bs, 1, 1]
        right_gripper_cur =  torch.tensor(np.stack([cur["follow_right_arm_joint_cur"][:,-1] for cur in batched_dict['joint_cur']])).unsqueeze(1) #[bs, 1, 1]
        obs_dict['agent_pos'] = torch.concat([obs_dict['agent_pos'], left_gripper_cur, right_gripper_cur], dim=-1)
    
    if use_joint_cur:
        if use_gripper_cur and use_joint_cur:
            assert False, f"You cannot use both gripper_cur and joint_cur at the same time!"
        left_joint_cur =  torch.tensor(np.stack([cur["follow_left_arm_joint_cur"][:, :] for cur in batched_dict['joint_cur']])) #[bs, 1, 7]
        right_joint_cur =  torch.tensor(np.stack([cur["follow_right_arm_joint_cur"][:, :] for cur in batched_dict['joint_cur']])) #[bs, 1, 7]
        # print(f"left_joint_cur: {left_joint_cur.shape}, right_joint_cur: {right_joint_cur.shape}")
        # print(f"obs_dict['agent_pos']: {obs_dict['agent_pos'].shape}")
        obs_dict['agent_pos'] = torch.concat([obs_dict['agent_pos'], left_joint_cur, right_joint_cur], dim=-1)

    actions = torch.as_tensor(actions[:, low_dim_obs_horizon:low_dim_obs_horizon + horizon, :action_dim])
    action_history = torch.as_tensor(np.stack(batched_dict['action_history']))
    if merge_cur_history:
        if use_joint_cur:
            assert False, "Not supported yet because history mixing is not implemented!"
        obs_dict['agent_pos'] = torch.concat([action_history, obs_dict['agent_pos']], dim=1) 

    item_dict = {
            'action': actions, 
            'obs': obs_dict, 
            "obs_goal": obs_dict_goal,
            "instruction": _pad_zero([
                torch.tensor([ord(c) for c in item], dtype=torch.int32)
                    for item in batched_dict['instruction']
            ], max_len=4096),
            "dataset_name": _pad_zero([
                torch.tensor([ord(c) for c in sample['dataset_name']], dtype=torch.int32)
                    for sample in batch
            ], max_len=100),
            'uid': _pad_zero([
                torch.tensor([ord(c) for c in uid], dtype=torch.int32) 
                    for uid in batched_dict['uid']
            ], max_len=200),
            'frame': torch.tensor(batched_dict['frame'])
        }

    if next_obs_dict:
        item_dict['next_obs'] = next_obs_dict

    if not merge_cur_history:
        item_dict['action_history'] = action_history

    if mask_keys:
        for mask_key in mask_keys:
            if len(batched_mask_dict[mask_key]) > 0:
                item_dict[mask_key] = torch.as_tensor(np.stack(batched_mask_dict[mask_key], axis=0))

    if 'tactile' in batched_dict:
        tactiles = torch.concatenate([torch.as_tensor(
                            np.stack([item[key][:low_dim_obs_horizon] for item in batched_dict['tactile']], axis=0)
                    ) for key in batched_dict['tactile'][0]
                ], dim=-1) # (bs, low_dim_obs_horizon, tactile_dim, tactile_num*tactile_space), (bs, 1, 3, 2*15)
        item_dict['tactile'] = tactiles


    return item_dict

def dreamer_collate_fn(batch,
                    #  obs_keys,
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
                     relative_action=False, # 是否是相对动作
                     add_noise=False, # 是否添加噪声
                     action_keys=None,
                     use_joint_action=False,
                     use_gripper_cur=False, # 是否使用gripper的力矩
                     action_shift=1,
                     ):
    # mask_type: mask_only return mask, mask_on_image return image(face_view) with mask

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    # for mask
    batched_mask_dict = {mask_key:[] for mask_key in mask_keys} if mask_keys else {}
    batched_dict = defaultdict(list)

    def _pad_zero(seq:List[torch.Tensor], max_len=None) -> torch.Tensor:
        max_len = max([len(item) for item in seq]) if max_len is None else max_len
        for i in range(len(seq)):
            seq[i] = torch.cat([seq[i], torch.zeros(max_len - len(seq[i]), dtype=torch.int32)])
        
        return torch.stack(seq)


    for sample in batch:
        # if sample2imginstruct is not None and sample['uid'] not in sample2imginstruct:
        #     continue
        add_origin = False

        robot_id = sample["uid"].split("_")[0]
        if robot_id in ["factory10002", "factory10003", "factory10004", "factory10005", 
                        "factory10006", "factory10007", "factory10008", "factory10009", 
                        "factory10010", "factory10011", "factory10012", "factory10013",
                        "factory10014",]:
            # print(sample["uid"], "add origin")
            add_origin = True

        actions = CollateHelper.stack_action(
                    sample["actions"], 
                    parse_head_action=parse_head_action,
                    fill_zero=True,
                    to_lie_algebra=to_lie_algebra,
                    relative_action=relative_action,
                    add_noise=add_noise,
                    action_keys=action_keys,
                    use_joint_action=use_joint_action,
                    add_origin=add_origin,
        )
        batched_dict['action'].append(actions)
    

        action_history = CollateHelper.stack_action(
                sample["action_histories"],
                parse_head_action=parse_head_action,
                fill_zero=True,
                to_lie_algebra=to_lie_algebra,
                relative_action=relative_action,
                add_noise=add_noise,
                action_keys=action_keys,
                use_joint_action=use_joint_action,
                add_origin=add_origin,
            )

        batched_dict['action_history'].append(action_history)

        obs_keys = list(sample["observations"].keys())
        observation = CollateHelper.stack_obs(sample["observations"], obs_keys)
        observation_goal = CollateHelper.stack_obs_goal(sample["observation_goal"], obs_keys)
        batched_dict['obs'].append(observation)
        batched_dict['obs_goal'].append(observation_goal)

        new_instruction = CollateHelper.stack_instruction(sample, sample2instruct)
        batched_dict['instruction'].append(new_instruction)
    
        if mask_type:
            assert mask_type in ['mask_only', 'mask_on_image', 'box_on_image'], f'{mask_type} is not support!'
            for mask_key in mask_keys:
                mask = sample[mask_key]
                if mask_type == 'mask_on_image': # not used now
                    mask = overlay_mask_on_image(image=observation['face_view'],  mask=mask)
                elif mask_type == 'box_on_image': # not used now
                    mask = overlay_box_on_image(image=observation['face_view'],  box=mask)
                elif mask_type == 'mask_only':
                    mask = overlay_masks(mask=mask, T=img_obs_horizon)
                batched_mask_dict[mask_key].append(mask)

        batched_dict['uid'].append(sample['uid'])
        batched_dict['frame'].append(sample['frame'])
        batched_dict['tactile'].append(sample['tactiles']) if 'tactiles' in sample else None
        batched_dict['joint_cur'].append(sample['joint_cur']) if 'joint_cur' in sample else None

    actions = np.stack(batched_dict['action'])
    # rearrange obs from (bs, img_obs_horizon, channel, height, width) to (bs, img_obs_horizon, height, width, channel)
    # first check if the last dimension is channel
    
    def resize_batch_time_images(images, target_size=(256, 256)):
        """Resize images with shape [B, T, H, W, C] to [B, T, target_H, target_W, C]"""
        B, T, H, W, C = images.shape
        
        # Reshape to [B*T, H, W, C] to apply transforms
        images = images.reshape(-1, H, W, C)
        
        # Convert to [B*T, C, H, W] for torchvision transforms
        images = images.permute(0, 3, 1, 2)
        
        # Apply resize
        resize_transform = TT.Resize(target_size, antialias=True)
        images = resize_transform(images)
        
        # Convert back to [B*T, H, W, C]
        images = images.permute(0, 2, 3, 1)
        
        # Reshape back to [B, T, H, W, C]
        images = images.reshape(B, T, target_size[0], target_size[1], C)
        
        return images

    obs_dict = {
        key : resize_batch_time_images(torch.as_tensor(
                np.stack([item[key] for item in batched_dict['obs']])[:, :img_obs_horizon, ...]
            )  )
            for key in obs_keys
    }
    obs_dict_goal = torch.as_tensor(np.array(batched_dict['obs_goal']))
    first_key = obs_keys[0]
    if obs_dict[first_key].shape[-1] != 3:
        obs_dict = {
            key: rearrange(obs_dict[key], 'b t c h w -> b t h w c') for key in obs_keys
        }
        obs_dict_goal = rearrange(obs_dict_goal, 'b c h w -> b h w c')
    
    obs_dict['agent_pos'] = torch.as_tensor(actions[:, :img_obs_horizon, :action_dim]) # (bs, low_dim_obs_horizon, action_dim)
    if use_gripper_cur:
        left_gripper_cur =  torch.tensor(np.stack([cur["follow_left_arm_joint_cur"][:,-1] for cur in batched_dict['joint_cur']])).unsqueeze(1) #[bs, 1, 1]
        right_gripper_cur =  torch.tensor(np.stack([cur["follow_right_arm_joint_cur"][:,-1] for cur in batched_dict['joint_cur']])).unsqueeze(1) #[bs, 1, 1]
        obs_dict['agent_pos'] = torch.concat([obs_dict['agent_pos'], left_gripper_cur, right_gripper_cur], dim=-1)

    actions = torch.as_tensor(actions[:, action_shift:action_shift + img_obs_horizon, :action_dim])
    action_history = torch.as_tensor(np.stack(batched_dict['action_history']))
    if merge_cur_history:
        obs_dict['agent_pos'] = torch.concat([action_history, obs_dict['agent_pos']], dim=1) 

    # 添加is_first, 不要最后一个维度
    is_first = torch.zeros(actions.shape[0], actions.shape[1])
    is_terminal = torch.zeros(actions.shape[0], actions.shape[1])
    is_first[:, 0] = 1
    # is_terminal[:, ] = 1

    item_dict = {
            'action': actions, 
            'face_view': obs_dict['right_wrist_view'][:,:, :, :, :],
            'agent_pose': obs_dict['agent_pos'],
            # 'obs': obs_dict, 
            # "obs_goal": obs_dict_goal,
            # "instruction": _pad_zero([
            #     torch.tensor([ord(c) for c in item], dtype=torch.int32)
            #         for item in batched_dict['instruction']
            # ], max_len=4096),
            # "dataset_name": _pad_zero([
            #     torch.tensor([ord(c) for c in sample['dataset_name']], dtype=torch.int32)
            #         for sample in batch
            # ], max_len=100),
            # 'uid': _pad_zero([
            #     torch.tensor([ord(c) for c in uid], dtype=torch.int32) 
            #         for uid in batched_dict['uid']
            # ], max_len=200),
            # 'frame': torch.tensor(batched_dict['frame']),
            'is_first': is_first,
            'is_terminal': is_terminal,
        }

    if not merge_cur_history:
        item_dict['action_history'] = action_history

    if mask_keys:
        for mask_key in mask_keys:
            if len(batched_mask_dict[mask_key]) > 0:
                item_dict[mask_key] = torch.as_tensor(np.stack(batched_mask_dict[mask_key], axis=0))
    
    # print(f"item_dict: {item_dict.keys()}")
    return item_dict


def chunk_collate_fn_mix(batch,
                     low_dim_obs_horizon,
                     img_obs_horizon,
                     horizon,
                     is_bi_mode=None,
                     sample2instruct=None,
                     to_lie_algebra=False,
                     action_keys=None,
                     ):
    batch = list(filter(lambda x: x is not None, batch))
    batched_actions, batched_obs, batched_instructions,batched_obs_goal = [], [], [], []
    batched_tactiles, batched_uid, batched_frame = [], [], []
    batched_action_history = []
    if is_bi_mode is not None:
        print("❗❗❗is_bi_mode is not used anymore")

    obs_keys = batch[0]["observations"].keys()
    for sample in batch:
        actions = CollateHelper.stack_action(sample["actions"], fill_zero=True, to_lie_algebra=to_lie_algebra, action_keys=action_keys)
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


def collate_wrapper(
        # obs_keys = ['face_view', 'left_wrist_view', 'right_wrist_view'],
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
        merge_image_history=False,
        relative_action=False,
        add_noise=False,
        action_keys=['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper'], 
        agent_pos_keys=None,
        use_joint_action=False,
        use_gripper_cur=False,
        use_joint_cur=False,
    ):
    '''
    sample2instruct is a dict, key is sample_id, value is list of instruct(str) or instruct(str)
    '''
    if collate_type == 'chunking':
        return functools.partial(chunk_collate_fn,
                        #   obs_keys=obs_keys,
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
                          merge_image_history=merge_image_history,
                          relative_action=relative_action,
                          add_noise=add_noise,
                          action_keys=action_keys,
                          agent_pos_keys=agent_pos_keys,
                          use_joint_action=use_joint_action,
                          use_gripper_cur=use_gripper_cur,
                          )
    elif collate_type == 'full':
        return functools.partial(full_collate_fn,
                        #   obs_keys=obs_keys,
                          action_dim=action_dim,
                          is_bi_mode=is_bi_mode,
                          parse_head_action=parse_head_action,
                          to_lie_algebra=to_lie_algebra,
                          action_keys=action_keys,
                          )
    elif collate_type == 'mix':
        return functools.partial(chunk_collate_fn_mix,
                                    low_dim_obs_horizon=low_dim_obs_horizon,
                                    img_obs_horizon=img_obs_horizon,
                                    horizon=horizon,
                                    sample2instruct=sample2instruct,
                                    to_lie_algebra=to_lie_algebra,
                                    action_keys=action_keys,
                                    )

    return None

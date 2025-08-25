from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

from torch.utils.data import Subset
import torch.distributed as dist
import random
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation
from x2robot_dataset.common.constants import ACTION_KEY_RANGES

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor],
        ignore_keys: List[str] = []
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if key not in ignore_keys:
            if isinstance(value, dict):
                result[key] = dict_apply(value, func)
            else:
                result[key] = func(value)
    return result

def decode_text(encoded_array:Union[torch.Tensor, np.ndarray]) -> str:
    '''
    Decode text from encoded array
    '''
    if isinstance(encoded_array, str):
        return encoded_array
    if len(encoded_array) == 0:
        return ''
    
    if isinstance(encoded_array, torch.Tensor):
        encoded_array = encoded_array.cpu().numpy()
    
    # 过滤掉 0 并转换为字符
    filtered_text = encoded_array[encoded_array != 0]

    return ''.join(map(chr, filtered_text))

def split_dataset_with_list(dataset, train_ratio=0.8):
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(dataset) * train_ratio)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset, train_indices, test_indices

def shuffle_sub_dataset(subset, seed):
    random.seed(seed)
    random.shuffle(subset.indices)
    
def print_rank0(*args, **kwargs):
    """
    Print the provided arguments only if this is rank 0.
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def convert_euler_to_Lang(euler_angle):
    """
    Convert Euler angles to Lang angle.
    
    Input: 
        euler_angle: pytorch tensor of shape [batch, 3] (Euler angles in radians)
    Output:
        lang_angle: numpy array of shape [batch] (Lang angles)
    """
    # Convert the PyTorch tensor to a NumPy array
    if isinstance(euler_angle, torch.Tensor):
        euler_angle_numpy = euler_angle.cpu().numpy()
    else:
        euler_angle_numpy = np.array(euler_angle)
    
    if len(euler_angle_numpy.shape) == 3:
        euler_angle_numpy = euler_angle_numpy.reshape(-1, 3)

    # Convert Euler angles to rotation matrix using scipy
    rotation_matrix = Rotation.from_euler('xyz', euler_angle_numpy).as_matrix()  # Shape: [batch, 3, 3]
    
    # Extract the relevant elements M00, M11, M22 for each rotation matrix
    M00 = rotation_matrix[:, 0, 0]  # First column, first row
    M11 = rotation_matrix[:, 1, 1]  # Second column, second row
    M22 = rotation_matrix[:, 2, 2]  # Third column, third row
    
    # Calculate the Lang angle
    lang_angle = np.arccos((M00 + M11 + M22 - 1) / 2)  # Shape: [batch]
    
    return lang_angle

def convert_6D_to_Lang(rotation_6d):
    """
    Convert 6D rotation to Lang angle. (Don't ask me why it is called Lang angle, quick coding)
    """
    if isinstance(rotation_6d, torch.Tensor):
        rotation_6d_numpy = rotation_6d.cpu().numpy()
    else:
        rotation_6d_numpy = np.array(rotation_6d)
    euler_angle = convert_6D_to_euler(rotation_6d_numpy)
    lang_angle = convert_euler_to_Lang(euler_angle)
    return lang_angle

def convert_euler_to_6D(euler_angle):
    """
    Convert euler angle to 6D rotation
    Input:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3] or [3]
    Output:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    """
    # TODO: find more elegent way
    # Convert euler angle to rotation matrix
    if len(euler_angle.shape) == 1:
        euler_angle = euler_angle.reshape(1, 3)
    rotation_matrix = Rotation.from_euler('xyz', euler_angle).as_matrix() # [horizon, 3, 3]
    # Convert rotation matrix to 6D rotation(first 2 columns of rotation matrix)
    rotation_6d = np.zeros((euler_angle.shape[0], 6))
    rotation_6d[:, :3] = rotation_matrix[:, :, 0]
    rotation_6d[:, 3:] = rotation_matrix[:, :, 1]
    assert rotation_6d.shape == (euler_angle.shape[0], 6), f"rotation_6d shape is not correct, you get {rotation_6d.shape}"
    return rotation_6d.squeeze() if len(euler_angle.shape) == 1 else rotation_6d

def convert_6D_to_euler(rotation_6d):
    """
    Convert 6D rotation to euler angle
    Input:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    Output:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3]
    """
    if rotation_6d.shape[0] == 6:
        rotation_6d = rotation_6d.reshape(1, 6)
    if len(rotation_6d.shape) == 3:
        rotation_6d = rotation_6d.reshape(-1, 6)
    # Convert 6D rotation to rotation matrix
    rotation_matrix = np.zeros((rotation_6d.shape[0], 3, 3))
    rotation_matrix[:, :, 0] = rotation_6d[:, :3]
    rotation_matrix[:, :, 1] = rotation_6d[:, 3:6]
    # get the third column of rotation matrix
    rotation_matrix[:, :, 2] = np.cross(rotation_matrix[:, :, 0], rotation_matrix[:, :, 1])
    assert rotation_matrix.shape == (rotation_6d.shape[0], 3, 3), "rotation_matrix shape is not correct"
    # Convert rotation matrix to euler angle
    euler_angle = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
    assert euler_angle.shape == (rotation_6d.shape[0], 3), "euler_angle shape is not correct"
    return euler_angle

def convert_xyzrpy_to_matrix(position, orientation, data_config):
    """
    Convert xyzrpy to matrix
    Input:
        postion: np.array [3]
        orientation: np.array [3] for euler angle or [6] for 6D representation
        data_config: X2...
    Output:
        configuration matrix: SE(3) np.array [4,4] 
    """
    configuration_matrix = np.eye(4)
    configuration_matrix[0:3, 3] = position
    if data_config.use_6D_rotation is True:
        orientation = convert_6D_to_euler(orientation)
    configuration_matrix[0:3, 0:3] = Rotation.from_euler('xyz', orientation).as_matrix()
    return configuration_matrix

def pose_to_transformation_matrix(pose, xyz_start_end_idx, rotation_start_end_idx, data_config):
    '''
    Input:
        pose: [action_dim], np.array
        xyz_start_end_idx: (,) tuple
        rotation_start_end_idx: (,) tuple
    Output:
        transformation_matrix: [4*4]
    '''
    position = pose[xyz_start_end_idx[0]: xyz_start_end_idx[1]]
    rotation = pose[rotation_start_end_idx[0]: rotation_start_end_idx[1]]
    transformation_matrix = convert_xyzrpy_to_matrix(position, rotation, data_config)
    return transformation_matrix

def absolute_pose_to_relative_pose(absolute_pose, pose_key, data_config, data_chunk_config, shape_mappings=ACTION_KEY_RANGES, drop_first_frame=False):
    """
    Definition of Pose: Position(xyz) + Orientation(rpy)
    Input: 
        absolute_pose: numpy array of shape [low_dim_obs_horizon+horizon, action_dim]
        pose_key: list of keys(strings) [num of action key]
        data_config: X2RDataProcessingConfig
        data_chunk_config: X2RDataChunkConfig
        drop_first_frame: whether drop the first frame for the returned value
        one_by_one_relative: whether to take the relative frame 相对于上一帧算relative action/相对于第一帧算relative action
    Output: 
        relative_pose: numpy array of shape [low_dim_obs_horizon+horizon, action_dim]
    """
    ### TODO: Not elegant enough
    # Construct action_dim idx according to data_config and data_chunk_config
    action_dim = absolute_pose.shape[-1]
    horizon = absolute_pose.shape[0]
    relative_pose = np.copy(absolute_pose) # copy grippers or other irrelevant data
    start_idx, end_idx = 0, 0
    action_dim_idx = {}
    for key in pose_key:
        end_idx += shape_mappings[key]['shape']
        action_dim_idx[key] = (start_idx, end_idx)
        start_idx = end_idx
    assert end_idx == action_dim, f"action_dim_idx is not correct, end_idx: {end_idx} != action_dim: {action_dim}"

    # Get the first frame of each arm
    left_position_key = [key for key in pose_key if 'cartesian' in key and 'left' in key]
    right_position_key = [key for key in pose_key if 'cartesian' in key and 'right' in key]
    left_orientation_key = [key for key in pose_key if 'rotation' in key and 'left' in key]
    right_orientation_key = [key for key in pose_key if 'rotation' in key and 'right' in key]

    assert len(left_position_key) == 1, f"there should be 1 left cartesian key, you get {len(left_position_key)}: {left_position_key}"
    assert len(right_position_key) == 1, f"there should be 1 right cartesian key, you get {len(right_position_key)}: {right_position_key}"
    assert len(left_orientation_key) == 1, f"there should be 1 left orientation key, you get {len(left_orientation_key)}: {left_orientation_key}"
    assert len(right_orientation_key) == 1, f"there should be 1 right orientation key, you get {len(right_orientation_key)}: {right_orientation_key}"

    left_position_key, right_position_key = left_position_key[0], right_position_key[0]
    left_orientation_key, right_orientation_key = left_orientation_key[0], right_orientation_key[0]

    left_position_idx = action_dim_idx[left_position_key]
    left_rotation_idx = action_dim_idx[left_orientation_key]
    right_position_idx = action_dim_idx[right_position_key]
    right_rotation_idx = action_dim_idx[right_orientation_key]
    
    ref_pose_left = pose_to_transformation_matrix(absolute_pose[0], left_position_idx, left_rotation_idx, data_config)
    ref_pose_left_inv = np.linalg.inv(ref_pose_left)
    ref_pose_right = pose_to_transformation_matrix(absolute_pose[0], right_position_idx, right_rotation_idx, data_config)
    ref_pose_right_inv = np.linalg.inv(ref_pose_right)

    relative_pose[0] = absolute_pose[0]
    for i in range(1, horizon):
        curr_pose_left = pose_to_transformation_matrix(absolute_pose[i], left_position_idx, left_rotation_idx, data_config)
        curr_pose_right = pose_to_transformation_matrix(absolute_pose[i], right_position_idx, right_rotation_idx, data_config)
        
        # compute relative matrix
        relative_pose_left = ref_pose_left_inv @ curr_pose_left
        relative_pose_right = ref_pose_right_inv @ curr_pose_right

        relative_pose[i, left_position_idx[0]:left_position_idx[1]] = relative_pose_left[:3, 3]
        relative_pose[i, right_position_idx[0]:right_position_idx[1]] = relative_pose_right[:3, 3]
        
        relative_orientation_left = Rotation.from_matrix(relative_pose_left[:3, :3]).as_euler('xyz')
        relative_orientation_right = Rotation.from_matrix(relative_pose_right[:3, :3]).as_euler('xyz')
        if data_config.use_6D_rotation is True:
            relative_orientation_left = convert_euler_to_6D(relative_orientation_left)
            relative_orientation_right = convert_euler_to_6D(relative_orientation_right)
        relative_pose[i, left_rotation_idx[0]:left_rotation_idx[1]] = relative_orientation_left
        relative_pose[i, right_rotation_idx[0]:right_rotation_idx[1]] = relative_orientation_right

        if data_config.one_by_one_relative is True:
            ref_pose_left = curr_pose_left
            ref_pose_right = curr_pose_right
            ref_pose_left_inv = np.linalg.inv(ref_pose_left)
            ref_pose_right_inv = np.linalg.inv(ref_pose_right)

    if drop_first_frame:
        return relative_pose[1:]
    else:
        return relative_pose

def relative_pose_to_absolute_pose(ref_pose, relative_pose, pose_key, data_config, data_chunk_config, shape_mappings=ACTION_KEY_RANGES, drop_first_frame=False):
    """
    Convert Delta EE to Real EE pose
    Input:
        ref_pose: np.array [1, action_dim]
        relative_pose: np.array [horizon, action_dim]
        pose_key: list for keys
        ... # TODO:以后再补，太困了orz zzZ
    Output:
        absolute_pose: np.array [horizon, action_dim]
    """
    if len(ref_pose.shape) == 1:
        ref_pose = ref_pose.reshape(1, ref_pose.shape[0])
    if len(relative_pose.shape) == 1:
        relative_pose = relative_pose.reshape(1, relative_pose.shape[0])

    ### TODO: Not elegant enough, too much redundent code
    horizon, action_dim = relative_pose.shape[0], relative_pose.shape[1]
    # assert ref_pose.shape == # TODO: Add shape check
    absolute_pose = np.zeros((horizon+1, action_dim))
    absolute_pose[0] = ref_pose[0]
    absolute_pose[1:] = relative_pose # copy grippers or other irrelevant data
    start_idx, end_idx = 0, 0
    action_dim_idx = {}
    for key in pose_key:
        end_idx += shape_mappings[key]['shape']
        action_dim_idx[key] = (start_idx, end_idx)
        start_idx = end_idx
    assert end_idx == action_dim, f"action_dim_idx is not correct, end_idx: {end_idx} != action_dim: {action_dim}"

    # get reference matrix
    position_key = [key for key in pose_key if 'cartesian' in key]
    orientation_key = [key for key in pose_key if 'rotation' in key]
    assert len(position_key) == 2, f"there should be 2 cartesian keys, you get {len(position_key)}: {position_key}"
    assert len(orientation_key)==2, f"there should be 2 orientation keys, you get {len(orientation_key)}: {orientation_key}"
    ref_position_left = ref_pose[0, action_dim_idx[position_key[0]][0]:action_dim_idx[position_key[0]][1]]
    ref_rotation_left = ref_pose[0, action_dim_idx[orientation_key[0]][0]:action_dim_idx[orientation_key[0]][1]]
    ref_pose_left = convert_xyzrpy_to_matrix(ref_position_left, ref_rotation_left, data_config)
    ref_position_right = ref_pose[0, action_dim_idx[position_key[1]][0]:action_dim_idx[position_key[1]][1]]
    ref_rotation_right = ref_pose[0, action_dim_idx[orientation_key[1]][0]:action_dim_idx[orientation_key[1]][1]]
    ref_pose_right = convert_xyzrpy_to_matrix(ref_position_right, ref_rotation_right, data_config)

    for i in range(0, horizon):
        relative_position_left = relative_pose[i, action_dim_idx[position_key[0]][0]:action_dim_idx[position_key[0]][1]]
        relative_rotation_left = relative_pose[i, action_dim_idx[orientation_key[0]][0]:action_dim_idx[orientation_key[0]][1]]
        relative_pose_left = convert_xyzrpy_to_matrix(relative_position_left, relative_rotation_left, data_config)
        absolute_pose_matrix_left = ref_pose_left @ relative_pose_left
        absolute_pose[i+1, action_dim_idx[position_key[0]][0]:action_dim_idx[position_key[0]][1]] = absolute_pose_matrix_left[:3, 3]
        orientation_left = Rotation.from_matrix(absolute_pose_matrix_left[:3, :3]).as_euler('xyz')
        if data_config.use_6D_rotation is True:
            orientation_left = convert_euler_to_6D(orientation_left)
        absolute_pose[i+1, action_dim_idx[orientation_key[0]][0]:action_dim_idx[orientation_key[0]][1]] = orientation_left

        relative_position_right = relative_pose[i, action_dim_idx[position_key[1]][0]:action_dim_idx[position_key[1]][1]]
        relative_rotation_right = relative_pose[i, action_dim_idx[orientation_key[1]][0]:action_dim_idx[orientation_key[1]][1]]
        relative_pose_right = convert_xyzrpy_to_matrix(relative_position_right, relative_rotation_right, data_config)
        absolute_pose_matrix_right = ref_pose_right @ relative_pose_right
        absolute_pose[i+1, action_dim_idx[position_key[1]][0]:action_dim_idx[position_key[1]][1]] = absolute_pose_matrix_right[:3, 3]
        orientation_right = Rotation.from_matrix(absolute_pose_matrix_right[:3, :3]).as_euler('xyz')
        if data_config.use_6D_rotation is True:
            orientation_right = convert_euler_to_6D(orientation_right)
        absolute_pose[i+1, action_dim_idx[orientation_key[1]][0]:action_dim_idx[orientation_key[1]][1]] = orientation_right

        if data_config.one_by_one_relative is True:
            ref_pose_left = absolute_pose_matrix_left
            ref_pose_right = absolute_pose_matrix_right

    if drop_first_frame:
        return absolute_pose[1:]
    else:
        return absolute_pose

def actions_to_relative(actions, add_noise=False, noise_scale=[0.05,0.05,0.05,0.05,0.05,0.05]):
    """Convert absolute actions to relative actions
    
    Args:
        actions: numpy array of shape [horizon, action_dim]
                action_dim=14, [left_arm(7), right_arm(7)]
                Each arm: [x,y,z,roll,pitch,yaw,gripper]
        add_noise: bool, whether to add noise to the first frame
        noise_scale: list of 6 numbers, scale of noise for [x,y,z,roll,pitch,yaw]
    
    Returns:
        relative_actions: numpy array of shape [horizon, action_dim]
    """
    horizon, _ = actions.shape
    relative_actions = np.zeros_like(actions)
    
    # 处理左臂和右臂
    for arm_idx in range(2):
        start_idx = arm_idx * 7
        
        # 获取第一帧的变换矩阵
        ref_pos = actions[0, start_idx:start_idx+3]
        ref_rot = Rotation.from_euler('xyz', actions[0, start_idx+3:start_idx+6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos
        
        # 如果需要加噪声，给第一帧加噪声
        if add_noise:
            noise = np.random.normal(scale=noise_scale, size=6)
            noisy_pos = ref_pos + noise[:3]
            noisy_rot = Rotation.from_euler('xyz', actions[0, start_idx+3:start_idx+6] + noise[3:])
            noisy_matrix = np.eye(4)
            noisy_matrix[:3, :3] = noisy_rot.as_matrix()
            noisy_matrix[:3, 3] = noisy_pos
            ref_matrix_inv = np.linalg.inv(noisy_matrix)
        else:
            ref_matrix_inv = np.linalg.inv(ref_matrix)
        
        # 计算所有帧的相对变换（包括第一帧）
        for i in range(horizon):
            # 当前帧的变换矩阵
            curr_pos = actions[i, start_idx:start_idx+3]
            curr_rot = Rotation.from_euler('xyz', actions[i, start_idx+3:start_idx+6])
            curr_matrix = np.eye(4)
            curr_matrix[:3, :3] = curr_rot.as_matrix()
            curr_matrix[:3, 3] = curr_pos
            
            # 计算相对变换
            relative_matrix = ref_matrix_inv @ curr_matrix
            
            # 提取相对位置和旋转
            relative_actions[i, start_idx:start_idx+3] = relative_matrix[:3, 3]
            relative_actions[i, start_idx+3:start_idx+6] = Rotation.from_matrix(
                relative_matrix[:3, :3]).as_euler('xyz')
            
            # Gripper保持不变
            relative_actions[i, start_idx+6] = actions[i, start_idx+6]
    
    return relative_actions

def relative_to_actions(relative_actions, start_pose, one_by_one_relative=False):
    """Convert relative actions back to absolute actions
    
    Args:
        relative_actions: numpy array of shape [horizon-1, action_dim]
                         从第二帧开始的相对位姿序列
                         action_dim=14, [left_arm(7), right_arm(7)]
                         Each arm: [x,y,z,roll,pitch,yaw,gripper]
        start_pose: numpy array of shape [action_dim]
                   第一帧的绝对位姿
    
    Returns:
        actions: numpy array of shape [horizon, action_dim]
                包含start_pose和转换后的绝对位姿序列
    """
    horizon = relative_actions.shape[0] + 1  # 加1是因为relative_actions不包含第一帧
    actions = np.zeros((horizon, relative_actions.shape[1]))
    
    # 设置第一帧为给定的start_pose
    actions[0] = start_pose
    
    # 处理左臂和右臂
    for arm_idx in range(2):
        start_idx = arm_idx * 7
        
        # 使用start_pose创建参考变换矩阵
        ref_pos = start_pose[start_idx:start_idx+3]
        ref_rot = Rotation.from_euler('xyz', start_pose[start_idx+3:start_idx+6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos
        
        # 从第二帧开始计算绝对位姿
        for i in range(horizon-1):  # horizon-1是relative_actions的长度
            # 当前相对位姿的变换矩阵
            relative_pos = relative_actions[i, start_idx:start_idx+3]
            relative_rot = Rotation.from_euler('xyz', relative_actions[i, start_idx+3:start_idx+6])
            relative_matrix = np.eye(4)
            relative_matrix[:3, :3] = relative_rot.as_matrix()
            relative_matrix[:3, 3] = relative_pos
            
            # 计算绝对变换
            abs_matrix = ref_matrix @ relative_matrix
            
            # 提取绝对位置和旋转，存储在第i+1帧
            actions[i+1, start_idx:start_idx+3] = abs_matrix[:3, 3]
            actions[i+1, start_idx+3:start_idx+6] = Rotation.from_matrix(
                abs_matrix[:3, :3]).as_euler('xyz')
            
            # Gripper保持不变
            actions[i+1, start_idx+6] = relative_actions[i, start_idx+6]

            # 当前帧的相对动作是相对于上一帧
            if one_by_one_relative == True:
                ref_matrix = abs_matrix
    
    return actions[1:] # 不包含start_pose

if __name__ == '__main__':
    def test_pose_conversion():
        """
        Test the conversion between absolute and relative poses to ensure consistency.
        """
        # 1. Define test parameters
        horizon = 10
        
        # Define keys for 3D (Euler) and 6D representations
        pose_keys_3d = [
            "follow_left_ee_cartesian_pos", "follow_left_ee_rotation", "follow_left_gripper",
            "follow_right_ee_cartesian_pos", "follow_right_ee_rotation", "follow_right_gripper"
        ]
        shape_mappings_3d = {
            "follow_left_ee_cartesian_pos": {'shape': 3}, "follow_left_ee_rotation": {'shape': 3}, "follow_left_gripper": {'shape': 1},
            "follow_right_ee_cartesian_pos": {'shape': 3}, "follow_right_ee_rotation": {'shape': 3}, "follow_right_gripper": {'shape': 1}
        }
        
        pose_keys_6d = [
            "follow_left_ee_cartesian_pos", "follow_left_ee_rotation_6D", "follow_left_gripper",
            "follow_right_ee_cartesian_pos", "follow_right_ee_rotation_6D", "follow_right_gripper"
        ]
        shape_mappings_6d = {
            "follow_left_ee_cartesian_pos": {'shape': 3}, "follow_left_ee_rotation_6D": {'shape': 6}, "follow_left_gripper": {'shape': 1},
            "follow_right_ee_cartesian_pos": {'shape': 3}, "follow_right_ee_rotation_6D": {'shape': 6}, "follow_right_gripper": {'shape': 1}
        }

        # 2. Generate valid random absolute pose data
        # Generate components and then assemble
        left_pos = np.random.rand(horizon, 3)
        left_euler = np.random.uniform(-np.pi, np.pi, size=(horizon, 3))
        left_gripper = np.random.rand(horizon, 1)
        
        right_pos = np.random.rand(horizon, 3)
        right_euler = np.random.uniform(-np.pi, np.pi, size=(horizon, 3))
        right_gripper = np.random.rand(horizon, 1)
        
        # Assemble 3D absolute pose
        absolute_pose_3d = np.concatenate([left_pos, left_euler, left_gripper, right_pos, right_euler, right_gripper], axis=1)
        
        # Assemble 6D absolute pose
        left_rot_6d = convert_euler_to_6D(left_euler)
        right_rot_6d = convert_euler_to_6D(right_euler)
        absolute_pose_6d = np.concatenate([left_pos, left_rot_6d, left_gripper, right_pos, right_rot_6d, right_gripper], axis=1)

        # Create a mock data_config class
        class MockDataConfig:
            def __init__(self, use_6D, one_by_one):
                self.use_6D_rotation = use_6D
                self.one_by_one_relative = one_by_one

        # 3. Loop through test cases
        for use_6d in [True, False]:
            for one_by_one in [True, False]:
                print(f"--- Testing with use_6D_rotation={use_6d}, one_by_one_relative={one_by_one} ---")
                
                data_config = MockDataConfig(use_6D=use_6d, one_by_one=one_by_one)
                data_chunk_config = None # Not used

                if use_6d:
                    test_absolute_pose = absolute_pose_6d
                    pose_keys = pose_keys_6d
                    shape_mappings = shape_mappings_6d
                else:
                    # When not using 6D, the functions expect euler angles and corresponding keys
                    test_absolute_pose = absolute_pose_3d
                    pose_keys = pose_keys_3d
                    shape_mappings = shape_mappings_3d

                # 4. Convert from absolute to relative
                relative_pose = absolute_pose_to_relative_pose(
                    test_absolute_pose.copy(), # Pass a copy to be safe
                    pose_key=pose_keys,
                    data_config=data_config,
                    data_chunk_config=data_chunk_config,
                    shape_mappings=shape_mappings,
                    drop_first_frame=True
                )

                # 5. Convert back from relative to absolute
                reconstructed_absolute_pose = relative_pose_to_absolute_pose(
                    ref_pose=test_absolute_pose[0],
                    relative_pose=relative_pose,
                    pose_key=pose_keys,
                    data_config=data_config,
                    data_chunk_config=data_chunk_config,
                    shape_mappings=shape_mappings,
                    drop_first_frame=True
                )
                
                # 6. Check for consistency
                original = test_absolute_pose[1:]
                
                if use_6d:
                    is_consistent = np.allclose(reconstructed_absolute_pose, original, atol=1e-6)
                else:
                    # For Euler angles, direct comparison can fail due to representation ambiguity.
                    # We must compare the rotation matrices they represent to verify consistency.
                    is_consistent = True
                    
                    # Define slices for clarity based on shape_mappings_3d
                    left_pos_slice = slice(0, 3)
                    left_rot_slice = slice(3, 6)
                    left_grip_slice = slice(6, 7)
                    right_pos_slice = slice(7, 10)
                    right_rot_slice = slice(10, 13)
                    right_grip_slice = slice(13, 14)

                    for i in range(len(original)):
                        o_pose = original[i]
                        r_pose = reconstructed_absolute_pose[i]

                        # Compare positions and grippers directly
                        if not (np.allclose(o_pose[left_pos_slice], r_pose[left_pos_slice], atol=1e-6) and
                                np.allclose(o_pose[right_pos_slice], r_pose[right_pos_slice], atol=1e-6) and
                                np.allclose(o_pose[left_grip_slice], r_pose[left_grip_slice], atol=1e-6) and
                                np.allclose(o_pose[right_grip_slice], r_pose[right_grip_slice], atol=1e-6)):
                            is_consistent = False
                            break

                        # Convert euler angles to rotation matrices and compare them
                        o_left_rotm = Rotation.from_euler('xyz', o_pose[left_rot_slice]).as_matrix()
                        r_left_rotm = Rotation.from_euler('xyz', r_pose[left_rot_slice]).as_matrix()
                        if not np.allclose(o_left_rotm, r_left_rotm, atol=1e-6):
                            is_consistent = False
                            break
                        
                        o_right_rotm = Rotation.from_euler('xyz', o_pose[right_rot_slice]).as_matrix()
                        r_right_rotm = Rotation.from_euler('xyz', r_pose[right_rot_slice]).as_matrix()
                        if not np.allclose(o_right_rotm, r_right_rotm, atol=1e-6):
                            is_consistent = False
                            break

                if is_consistent:
                    print("✅ Test PASSED: Poses are consistent.")
                else:
                    print("❌ Test FAILED: Poses are NOT consistent.")
                    # Optional: print details for debugging
                    # print("Original absolute pose (from frame 1):")
                    # print(original)
                    # print("Reconstructed absolute pose (from frame 1):")
                    # print(reconstructed_absolute_pose)
                    # print("Difference:")
                    # print(original - reconstructed_absolute_pose)

    test_pose_conversion()

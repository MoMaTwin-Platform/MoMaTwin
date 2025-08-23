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

def relative_to_actions(relative_actions, start_pose):
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
    
    return actions[1:] # 不包含start_pose

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


def remove_outliers(data, threshold=3):
    """
    使用Z-score方法移除异常值。
    
    参数:
        data: 输入数据数组
        threshold: Z-score阈值，默认为3
        
    返回:
        过滤后的数据
    """
    # # 如果数据点太少或全为相同值，直接返回原始数据
    # if len(data) < 3 or np.all(data == data[0]):
    #     return data.copy()

    # # 计算标准差，避免灾难性抵消
    # std = np.std(data)
    # if std < 1e-10:  # 如果标准差接近0，直接返回原始数据
    #     return data.copy()
        
    # # 计算Z-score
    # try:
    #     z_scores = np.abs(stats.zscore(data))
    # except:
    #     # 如果无法计算Z-score（例如所有值相同），直接返回原始数据
    #     return data.copy()
        
    filtered_data = data.copy()
    
    # # 找出异常值
    # mask = z_scores > threshold
    
    # # 如果异常值太多，可能是数据本身就有较大波动，保持原样
    # if np.sum(mask) > len(data) * 0.4:  # 如果超过40%的数据被标记为异常
    #     return data.copy()
        
    # # 标记异常值为NaN
    # filtered_data[mask] = np.nan
    
    # # 检查是否所有值都变成了NaN
    # if np.all(np.isnan(filtered_data)):
    #     return data.copy()
        
    # # 用插值填充NaN值
    # nan_mask = np.isnan(filtered_data)
    
    # # 确保有非NaN值可以用于插值
    # if np.any(~nan_mask):
    #     filtered_data[nan_mask] = np.interp(
    #         np.flatnonzero(nan_mask), 
    #         np.flatnonzero(~nan_mask), 
    #         filtered_data[~nan_mask]
    #     )
    
    return filtered_data

def remove_jumps(data, threshold=1.0):
    """
    检测并修正数据中的突然跳变。
    
    参数:
        data: 输入数据数组
        threshold: 跳变检测阈值，默认为1.0
        
    返回:
        修正后的数据
    """
    # 如果数据点太少，无法有效检测跳变，直接返回原始数据
    if len(data) < 3:
        return data.copy()
        
    result = data.copy()
    
    # 计算相邻点之间的差值
    try:
        diffs = np.abs(np.diff(result))
    except:
        # 如果无法计算差值，直接返回原始数据
        return data.copy()
        
    # 找出大于阈值的跳变点
    jump_indices = np.where(diffs > threshold)[0]
    
    # 如果跳变点太多，说明数据本身波动较大，保持原样
    if len(jump_indices) > len(data) * 0.3:  # 如果超过30%的点被标记为跳变
        return data.copy()
    
    # 处理每个跳变点
    for idx in jump_indices:
        # 用前后点的平均值替换跳变
        if idx > 0 and idx < len(result) - 1:
            # 取跳变前后的平均值
            result[idx+1] = (result[idx] + result[idx+2]) / 2
        elif idx == len(result) - 2:  # 如果是倒数第二个点
            result[idx+1] = result[idx]  # 用前一个点的值替换
    
    return result

def smooth_data(data, window_length=None, polyorder=3, iterations=1, strong_smooth=False):
    """
    使用Savitzky-Golay滤波器平滑数据。
    
    参数:
        data: 输入数据数组
        window_length: 窗口长度，如果未指定则自动计算
        polyorder: 多项式阶数，默认为3
        iterations: 平滑迭代次数，默认为1
        strong_smooth: 是否使用强平滑模式，默认为False
        
    返回:
        平滑后的数据
    """
    # 确保数据至少有3个点
    if len(data) < 3:
        return data.copy()
        
    # 计算合适的窗口长度
    if window_length is None:
        if strong_smooth:
            # 强平滑模式下使用更大的窗口
            window_length = min(51, len(data) - 1)
        else:
            window_length = min(21, len(data) - 1)
        
    # 确保窗口长度为奇数且小于等于数据长度
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:  # 确保窗口长度为奇数
        window_length -= 1
    
    # 确保窗口长度至少为3
    window_length = max(3, window_length)
    
    # 如果数据长度小于窗口长度，使用高斯滤波
    if window_length >= len(data):
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)
    
    # 在强平滑模式下使用更低的多项式阶数
    if strong_smooth:
        polyorder = min(2, polyorder)
    
    # 确保多项式阶数小于窗口长度
    polyorder = min(polyorder, window_length - 1)
    
    smooth_data_result = data.copy()
    
    try:
        # 应用多次平滑
        for _ in range(iterations):
            smooth_data_result = savgol_filter(smooth_data_result, window_length, polyorder)
            
        # 如果需要强平滑，再应用高斯滤波
        if strong_smooth:
            smooth_data_result = gaussian_filter1d(smooth_data_result, sigma=2.0)
            
        return smooth_data_result
        
    except Exception as e:
        # 如果savgol_filter失败，使用高斯滤波作为备选
        print(f"Savgol filter failed: {e}, using Gaussian filter instead")
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)
    
def process_car_pose_to_base_velocity(car_pose, 
                                    outlier_threshold=3, 
                                    jump_threshold=1.0, 
                                    smooth_iterations=3, 
                                    strong_smooth=True):
    """
    🎯 处理car_pose数据，转换为本体坐标系base_velocity_decomposed，与batch_process_json_data.py完全一致。
    包含完整的异常值处理、角度unwrap、过滤和平滑处理，以及本体坐标系速度计算。
    
    参数:
        car_pose: 输入的car_pose数据，shape: (n, 3) [x, y, angle]
        outlier_threshold: 异常值检测阈值，默认为3
        jump_threshold: 跳变检测阈值，默认为1.0
        smooth_iterations: 平滑迭代次数，默认为3
        strong_smooth: 是否使用强平滑模式，默认为True
        
    返回:
        dict: 包含处理后的数据
            - 'base_velocity_decomposed': shape (n, 3) [vx_body, vy_body, vyaw] (本体坐标系)
            - 'valid': bool, 是否有效（通过速度范围检查）
    """
    # 定义速度限制（与data_analysis_filter.py中完全一致）
    velocity_limits = {
        'vx': {'min': -0.5, 'max': 0.5},
        'vy': {'min': -0.5, 'max': 0.5}, 
        'vyaw': {'min': -1.6, 'max': 1.6}
    }
    
    # 处理空数据或单点数据
    if len(car_pose) == 0:
        return {
            'base_velocity_decomposed': np.zeros((0, 3)),
            'valid': False
        }

    if len(car_pose) == 1:
        return {
            'base_velocity_decomposed': np.zeros((1, 3)),
            'valid': True  # 单点数据认为是有效的
        }

    # 🎯 步骤1: 提取位置和角度数据，并进行角度展开
    x_values = car_pose[:, 0].copy()
    y_values = car_pose[:, 1].copy()
    angle_values = car_pose[:, 2].copy()

    # 🎯 角度展开处理，避免跳变（移到函数内部）
    angle_values_unwrapped = np.unwrap(angle_values)

    # 🎯 步骤2-4: 异常值处理、跳变修正、平滑处理
    # 异常值处理
    x_filtered = remove_outliers(x_values, outlier_threshold)
    y_filtered = remove_outliers(y_values, outlier_threshold)
    angle_filtered = remove_outliers(angle_values_unwrapped, outlier_threshold)

    # 跳变修正
    x_filtered = remove_jumps(x_filtered, jump_threshold)
    y_filtered = remove_jumps(y_filtered, jump_threshold)
    angle_filtered = remove_jumps(angle_filtered, jump_threshold)

    # 平滑处理
    window_length = min(51 if strong_smooth else 21, len(x_filtered) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)

    x_smooth = smooth_data(x_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                          iterations=smooth_iterations, strong_smooth=strong_smooth)
    y_smooth = smooth_data(y_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                          iterations=smooth_iterations, strong_smooth=strong_smooth)
    angle_smooth = smooth_data(angle_filtered, window_length, polyorder=2 if strong_smooth else 3, 
                              iterations=smooth_iterations, strong_smooth=strong_smooth)

    # 🎯 步骤5: 使用与data_processor.py完全一致的本体坐标系速度计算方法
    dt = 1/20  # 20Hz采样频率
    
    # 计算全局位移
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    angle_diff = np.diff(angle_smooth)
    
    # 获取当前帧的角度（用于坐标变换）
    current_theta = angle_smooth[:-1]  # shape: (n-1,)
    
    # 🎯 坐标变换：从全局坐标系到本体坐标系（与data_processor.py一致）
    cos_theta = np.cos(current_theta)
    sin_theta = np.sin(current_theta)
    
    # 🎯 计算本体坐标系下的速度
    vx_body = (x_diff * cos_theta + y_diff * sin_theta) / dt   # 前进速度（本体坐标系）
    vy_body = (-x_diff * sin_theta + y_diff * cos_theta) / dt  # 左侧速度（本体坐标系）
    vyaw = angle_diff / dt  # 角速度
    
    # 在开头添加零速度以匹配原始数据长度
    vx_array = np.concatenate([[0], vx_body])
    vy_array = np.concatenate([[0], vy_body])
    vyaw_array = np.concatenate([[0], vyaw])
    
    base_velocity_decomposed = np.stack([vx_array, vy_array, vyaw_array], axis=1)
    
    # 🎯 步骤6: 速度范围检查（与data_analysis_filter.py一致）
    valid = True
    
    # # 检查每个速度分量是否在允许范围内
    # for vx_val in vx_body:
    #     if vx_val < velocity_limits['vx']['min'] or vx_val > velocity_limits['vx']['max']:
    #         valid = False
    #         break
    
    # if valid:  # 只有vx通过检查才继续检查vy
    #     for vy_val in vy_body:
    #         if vy_val < velocity_limits['vy']['min'] or vy_val > velocity_limits['vy']['max']:
    #             valid = False
    #             break
    
    # if valid:  # 只有vx和vy都通过检查才继续检查vyaw
    #     for vyaw_val in vyaw:
    #         if vyaw_val < velocity_limits['vyaw']['min'] or vyaw_val > velocity_limits['vyaw']['max']:
    #             valid = False
    #             break

    return {
        'base_velocity_decomposed': base_velocity_decomposed,
        'valid': valid
    }
    
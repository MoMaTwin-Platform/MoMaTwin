# data preprocessing for x2robot raw data   

import numpy as np
from collections import defaultdict

import imageio
import json
import os
import tqdm
import random
import torch
import re

from x2robot_dataset.common.data_utils import print_rank0

def check_files(video_file_groups,
                cam_list,
                detect_phase_switch=False,
                detect_motion=False,
                report_file=None):
    # 使用集合提升查找效率
    invalid_files, phase_switch = set(), []
    
    invalid_keys = ['vague_sample', 'out_of_3_sigma', 'gripper_fast_oscillations', 
                   'select_low_quality', "select_not_standard"]
    
    if detect_motion:
        invalid_keys.append('No_action')
    
    phase_switch_keys = ['rot_out_of_range','Switch_Over_Range'] if detect_phase_switch else []
    if not detect_phase_switch:
        invalid_keys.extend(phase_switch_keys)

    # 预处理摄像头列表后缀检查
    cam_suffixes = {f'/{cam_name}.mp4' for cam_name in cam_list}

    # 预处理报告文件
    if report_file and os.path.exists(report_file):  # 添加文件存在性检查
        with open(report_file, 'r') as f:
            report = json.load(f)
        for key in report:
            if key in phase_switch_keys:
                phase_switch.extend(report[key])
            elif key in invalid_keys:
                invalid_files.update(report[key])  # 直接转为集合

    valid_video_file_groups = []
    for video_file in video_file_groups:
        base_name = os.path.basename(video_file)
        
        # 提前进行无效文件检查
        if base_name in invalid_files:
            print_rank0('Warning: invalid file ', base_name)
            continue
            
        # 批量检查所有文件路径
        required_files = [
            os.path.join(video_file, f"{base_name}.json"),
            *[video_file + suf for suf in cam_suffixes]
        ]
        
        if not all(os.path.exists(f) for f in required_files):
            continue  # 自动跳过缺失文件的情况
        
        valid_video_file_groups.append(video_file)

    if phase_switch:
        print_rank0('Warning: The following trajectories would be smoothed:', phase_switch)

    return valid_video_file_groups, phase_switch

_CAM_FILE_MAPPING = {
    'face_view': 'faceImg.mp4',
    'left_wrist_view': 'leftImg.mp4',
    'right_wrist_view': 'rightImg.mp4',
    'face_view_aug': 'faceImg_aug.mp4', # 用于模型的增强训练
    'left_wrist_view_aug': 'leftImg_aug.mp4',
    'right_wrist_view_aug': 'rightImg_aug.mp4'
}
# 双目摄像头
_CAM_BINOCULAR_FILE_MAPPING = {
    'head_left_view': 'head_leftImg.mp4',
    'head_right_view': 'head_rightImg.mp4',
    'left_wrist_view': 'leftImg.mp4',
    'right_wrist_view': 'rightImg.mp4'
}
_CAM_FILE_MAPPING_ALL = _CAM_FILE_MAPPING | _CAM_BINOCULAR_FILE_MAPPING

_CAM_MAPPING={'faceImg':'face_view',
             'leftImg':'left_wrist_view',
             'rightImg':'right_wrist_view'}

# 目前背景增强只对faceImg进行增强
_CAM_MAPPING_BACKGROUND_AUG = {'faceImg_aug':'face_view',
             'leftImg':'left_wrist_view',
             'rightImg':'right_wrist_view'}

_CAM_MAPPING_LIGHT_AUG = {'faceImg_light':'face_view',
             'leftImg_light':'left_wrist_view',
             'rightImg_light':'right_wrist_view'}

_CAM_BINOCULAR_MAPPING={'head_leftImg':'head_left_view',
             'head_rightImg':'head_right_view',
             'leftImg':'left_wrist_view',
             'rightImg':'right_wrist_view'}

def process_video(video_path, sampling_freq=20):
    reader = imageio.get_reader(video_path, 'ffmpeg')

    fps = reader.get_meta_data()['fps']

    if fps == sampling_freq:
        frame_arrays = np.array([frame for frame in reader])
    else:
        interval = int(round(fps / sampling_freq))

        frames = []

        for i, frame in enumerate(reader):
            if i % interval == 0:
                frames.append(frame)
        frame_arrays = np.array([np.array(frame) for frame in frames])

    reader.close()

    return frame_arrays

def process_videos(file_path, obs_key=None):
    camera_dict = _CAM_FILE_MAPPING_ALL
    if obs_key is not None:
        cam_list = []
        cam_desc = []
        for key in obs_key:
            if key in camera_dict:
                cam_list.append(os.path.join(file_path, camera_dict[key]))
                cam_desc.append(key)
    else:
        cam_list, cam_desc = [], []
        for name,cam_file in camera_dict.items():
            if not os.path.exists(os.path.join(file_path, cam_file)):
                continue
            else:
                cam_list.append(os.path.join(file_path, cam_file))
                cam_desc.append(name)

    video = {name:process_video(video_path) for name,video_path in zip(cam_desc, cam_list)}

    return video

# 统一的action key映射：模型key -> 原始数据key
_ACTION_KEY_FULL_MAPPING = {
    # ARX系列
    'follow_right_arm_joint_pos': 'follow_right_joint_pos',
    'follow_right_arm_joint_dev': 'follow_right_joint_dev',
    'follow_right_arm_joint_cur': 'follow_right_joint_cur',
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_cartesian_pos_relative': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_ee_rotation_relative': 'follow_right_rotation',
    'follow_right_ee_rotation_6D': 'follow_right_rotation',
    'follow_right_ee_rotation_6D_relative': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'master_right_arm_joint_pos': 'master_right_joint_pos',
    'master_right_arm_joint_dev': 'master_right_joint_dev',
    'master_right_arm_joint_cur': 'master_right_joint_cur',
    'master_right_ee_cartesian_pos': 'master_right_position',
    'master_right_ee_cartesian_pos_relative': 'master_right_position',
    'master_right_ee_rotation': 'master_right_rotation',
    'master_right_ee_rotation_relative': 'master_right_rotation',
    'master_right_ee_rotation_6D': 'master_right_rotation',
    'master_right_ee_rotation_6D_relative': 'master_right_rotation',
    'master_right_gripper': 'master_right_gripper',
    'follow_left_arm_joint_pos': 'follow_left_joint_pos',
    'follow_left_arm_joint_dev': 'follow_left_joint_dev',
    'follow_left_arm_joint_cur': 'follow_left_joint_cur',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_cartesian_pos_relative': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_ee_rotation_relative': 'follow_left_rotation',
    'follow_left_ee_rotation_6D': 'follow_left_rotation',
    'follow_left_ee_rotation_6D_relative': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper',
    'master_left_arm_joint_pos': 'master_left_joint_pos',
    'master_left_arm_joint_dev': 'master_left_joint_dev',
    'master_left_arm_joint_cur': 'master_left_joint_cur',
    'master_left_ee_cartesian_pos': 'master_left_position',
    'master_left_ee_rotation': 'master_left_rotation',
    'master_left_ee_cartesian_pos_relative': 'master_left_position',
    'master_left_ee_rotation_relative': 'master_left_rotation',
    'master_left_ee_rotation_6D': 'master_left_rotation',
    'master_left_ee_rotation_6D_relative': 'master_left_rotation',
    'master_left_gripper': 'master_left_gripper',
    
    # JAKA系列 - 添加原始数据映射
    'follow_left_ee_cartesian_pos_jaka': 'follow_left_position',
    'follow_right_ee_cartesian_pos_jaka': 'follow_right_position',
    'follow_left_ee_rotation_jaka': 'follow_left_rotation',
    'follow_right_ee_rotation_jaka': 'follow_right_rotation',
    'follow_left_arm_joint_pos_jaka': 'follow_left_joint_pos',
    'follow_right_arm_joint_pos_jaka': 'follow_right_joint_pos',
    'follow_left_arm_joint_cur_jaka': 'follow_left_joint_cur',
    'follow_right_arm_joint_cur_jaka': 'follow_right_joint_cur',
    
    # 乌龟+移动支架系列
    'velocity_decomposed': 'velocity_decomposed',
    'car_pose': 'car_pose',
    "height":"lifting_mechanism_position",
    'head_rotation': 'head_rotation',

    # 手部控制
    'follow_left_hand_joint_pos': 'follow_left_hand_joint_pos',
    'follow_left_hand_joint_dev': 'follow_left_hand_joint_dev',
    'follow_right_hand_joint_pos': 'follow_right_hand_joint_pos',
    'follow_right_hand_joint_dev': 'follow_right_hand_joint_dev',
    
    # 夹爪力控 - 从arm_joint_cur的最后一个关节提取
    'follow_left_gripper_cur': 'follow_left_joint_cur[-1]',
    'follow_right_gripper_cur': 'follow_right_joint_cur[-1]',

}

_ACTION_KEY_FULL_MAPPING_INV = {v:k for k,v in _ACTION_KEY_FULL_MAPPING.items()}

def process_action(file_path, raw_key2model_key=_ACTION_KEY_FULL_MAPPING_INV, filter_angle_outliers=True):
    # 判断是否是文件路径（字符串）
    if isinstance(file_path, str):
        file_name = os.path.basename(file_path)
        action_path = os.path.join(file_path, f"{file_name}.json")
        
        with open(action_path, 'r') as file:
            actions = json.load(file)
    else:
        # 假设传入的是已加载的字典对象（如yaml.load的结果）
        actions = file_path

    data = actions.get('data', [])  # 使用 .get() 防止 key error
    
    # 分离普通映射和特殊映射
    normal_mappings = {}
    special_mappings = {}
    
    # 正则表达式匹配索引操作：[数字] 或 [-数字]
    index_pattern = re.compile(r'^(.+)\[(-?\d+)\]$')
    
    for raw_key, model_key in raw_key2model_key.items():
        if isinstance(raw_key, str):
            match = index_pattern.match(raw_key)
            if match:
                # 特殊映射：如 'follow_left_joint_cur[0]' -> 'follow_left_gripper_cur'
                base_raw_key = match.group(1)
                index = int(match.group(2))
                special_mappings[base_raw_key] = {
                    'model_key': model_key,
                    'index': index,
                    'original_key': raw_key
                }
            else:
                # 普通映射
                normal_mappings[raw_key] = model_key
        else:
            # 普通映射
            normal_mappings[raw_key] = model_key
    # 收集所有需要的原始数据key（包括特殊映射需要的base_raw_key）
    needed_raw_keys = set(normal_mappings.keys())
    needed_raw_keys.update(special_mappings.keys())
    
    # 从原始数据中收集所有需要的数据
    raw_data = defaultdict(list)
    for action in data:
        for key, val in action.items():
            if key in needed_raw_keys:
                raw_data[key].append(val)
    
    # 转换为numpy数组
    raw_data = {k: np.array(v, dtype=np.float32) for k, v in raw_data.items()}
    
    # 初始化结果字典
    trajectories = {}
    
    # 处理普通映射
    for raw_key, model_key in normal_mappings.items():
        if raw_key in raw_data:
            trajectories[model_key] = raw_data[raw_key]
    
    # 处理特殊映射：根据索引从数组中提取数据
    for base_raw_key, mapping_info in special_mappings.items():
        model_key = mapping_info['model_key']
        index = mapping_info['index']
        original_key = mapping_info['original_key']
        
        if base_raw_key in raw_data:
            # 直接从原始数据中根据索引提取数据
            source_data = raw_data[base_raw_key]
            
            if source_data.ndim == 2 and source_data.shape[1] > 0:
                # 使用numpy的高级索引，支持负索引
                if index >= 0:
                    if index < source_data.shape[1]:
                        trajectories[model_key] = source_data[:, index:index+1] # 保持2D shape
                    else:
                        print_rank0(f"Warning: Index {index} out of range for {base_raw_key} (shape: {source_data.shape})")
                else:
                    # 负索引处理：先转换为正索引
                    if abs(index) <= source_data.shape[1]:
                        positive_index = source_data.shape[1] + index  # 转换为正索引
                        trajectories[model_key] = source_data[:, positive_index:positive_index+1] # 保持2D shape
                    else:
                        print_rank0(f"Warning: Negative index {index} out of range for {base_raw_key} (shape: {source_data.shape})")
            else:
                print_rank0(f"Warning: Cannot extract {model_key} from {base_raw_key}, invalid shape: {source_data.shape}")
                
        else:
            print_rank0(f"Warning: {base_raw_key} not found in raw_data for extracting {model_key}")
            print_rank0(f"Available raw_data keys: {list(raw_data.keys())}")
    
    if filter_angle_outliers:
        trajectories = smooth_action(trajectories)
    
    return trajectories


_TAC_FILE_MAPPING = {
    'left': 'left_tactile',
    'right': 'right_tactile'
}
_TAC_FILE_MAPPING_V2 = {
    'tactile_data_left': 'left_tactile',
    'tactile_data_right': 'right_tactile'
}
_HEAD_ACTION_MAPPING = {
    'head_actions': 'head_rotation'
}
_HEAD_ACTION_MAPPING_INV = {v:k for k,v in _HEAD_ACTION_MAPPING.items()}
def process_tactility_old(file_path,
                      tac_key_mapping=_TAC_FILE_MAPPING,
                      ):
    tac_path = os.path.join(file_path, "touch.json")
    if not os.path.exists(tac_path):
        return None
    # Open the JSON file
    trajectories = defaultdict(lambda:[])
    with open(tac_path, 'r') as file:
        tacs = json.load(file)
        for tac in tacs['data']:
            for key, val in tac.items():
                val = np.array(val).reshape(-1, 3).T #[[0,3,6...],[1,4,7...],[2,5,8...]]
                trajectories[tac_key_mapping[key]].append(val) # (力维度，x轴维度*y轴维度)

        trajectories  = {key:np.array(val, dtype=np.int16) for key,val in trajectories.items()}# (n, 3, 3, 5)
        return trajectories

def process_tactility(file_path, tac_key_mapping=_TAC_FILE_MAPPING_V2):
    action_path = os.path.join(file_path, f"{file_path.split('/')[-1]}.json")
    trajectories = defaultdict(lambda:[])
    with open(action_path, 'r') as file:
        actions = json.load(file)
        for action in actions['data']:
            for key, val in action.items():
                if key in tac_key_mapping.keys():
                    trajectories[tac_key_mapping[key]].append(np.array(val))
    return trajectories

import pandas as pd
def smooth_action(action):
    def _filter(traj, threshold = 3, alpha = 0.05, window=10):
        # Convert to pandas Series but preserve the original dtype
        orig_dtype = traj.dtype
        data = pd.Series(traj)
        derivatives = np.diff(data)

        spike_indices = np.where(abs(derivatives) > threshold)[0]
        if len(spike_indices) > 0:
            ema = data.ewm(alpha=alpha, adjust=True).mean()
            
            # Fix: Ensure the slice indices are within bounds
            start_idx = max(0, spike_indices[0] - window)
            end_idx = min(len(data), spike_indices[-1] + window + 1)
            
            # Get the corresponding segment from the EMA
            modified_seg = ema.iloc[start_idx:end_idx]
            
            # Ensure the lengths match before assignment and explicitly convert to the original dtype
            if len(modified_seg) > 0:
                # Convert values back to the original dtype before assignment
                data.iloc[start_idx:end_idx] = modified_seg.values.astype(orig_dtype)
                
        return data.to_numpy().astype(orig_dtype)  # Ensure we return the same dtype

    for key in ['follow_right_ee_rotation', 'follow_left_ee_rotation']:
        if key in action:  # Check if the key exists in the action dictionary
            try:
                # Process each dimension separately while preserving dtype
                orig_dtype = action[key].dtype
                filtered_traj = np.stack([_filter(action[key][:,i]) for i in range(3)], axis=1)
                if not np.isnan(filtered_traj).any():
                    action[key] = filtered_traj.astype(orig_dtype)  # Ensure consistent dtype
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not smooth {key} due to error: {e}")
    
    return action

def is_stationary(frame_data, threshold=0.01):
    """检查当前帧是否静止，适应不同维度的数据"""
    stationary = True
    for data in frame_data:
        if data.ndim == 1:
            diffs = np.abs(np.diff(data))
        else:  # data.ndim == 2
            diffs = np.linalg.norm(np.diff(data, axis=0), axis=1)
        if np.any(diffs >= threshold):
            stationary = False
            break
    return stationary

def trim_stationary_ends(action_data, threshold=0.01):
    """裁剪所有动作数据的静止头尾"""
    keys = list(action_data.keys())
    n_frames = len(action_data[keys[0]])
    
    start, end = 0, n_frames

    # 检查开头的静止状态
    while start < end - 1:
        frame_data = [action_data[key][start:start+2] for key in keys]
        if not is_stationary(frame_data, threshold):
            break
        start += 1

    # 检查结尾的静止状态
    while end > start + 1:
        frame_data = [action_data[key][end-2:end] for key in keys]
        if not is_stationary(frame_data, threshold):
            break
        end -= 1

    # 裁剪静止的头尾
    trimmed_data = {key: action_data[key][start:end] for key in keys}
    return trimmed_data, (start, end)

TASK_DESC_MAPPING = {
    'sausage': 'cut the sausage with the knife',
    'cucumber': 'cut cucumber with the knife',
    'juice': 'open the juice powder can, and use the spoon to add the powder to the cup',
    'clean': 'use sponge to clean the table',
    'salad': 'add sauce to the bowl containing cucumbers',
    'flower': 'watering the flower with the watering can',
    'water-flowers': 'watering the flowers with the watering can',
    'clothes': 'fold the clothes',
    'water': 'pour water from kettle into the cup containing juice powder and stir with a spoon to make juice',
    'pour-water': 'pour water from kettle into the cup containing juice powder and stir with a spoon to make juice',
    'wastesorting': 'put dry trash in dry trash bin, and put wet trash in wet trash bin',
}

def process_instruction(file_path):
    task_name = file_path.split('/')[-2]
    for key, val in TASK_DESC_MAPPING.items():
        if key in task_name:
            return {
                "text_en":[val]
            }

import requests
import re
def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None

def translate_zh2en(zh_text, default_url=None):
    # if translations is not None and zh_text in translations.keys():
    #     return translations[zh_text]
    if isinstance(default_url, dict):
        # default_url是一个包含{zh_text: [en_text1, en_text2, ...]}的字典
        if zh_text in default_url.keys():
            if isinstance(default_url[zh_text], list):
                return random.choice(default_url[zh_text])
            else:
                return default_url[zh_text]
        else:
            return zh_text

    if default_url is None:
        return ''

    url = default_url #'http://127.0.0.1:5000/predict'
    data = {'prompt': zh_text}
    response = requests.post(url, json=data)
   
    retry = 0
    while response.status_code == 200: 
        text = response.json()['response']
        if contains_chinese(text):
            print_rank0(data, '\n', text)
            response = requests.post(url, json=data)
        else:
            return text

        if retry > 10:
            print_rank0(f'No response from {url}')
            return ''
        retry += 1



class LabelData:
    def __init__(self, file_pathes, default_url=None):
        if isinstance(file_pathes, str):
            file_pathes = [file_pathes]
        self.data = {}
        for file_path in file_pathes:
            data = LabelData.read_label_data(file_path, default_url)
            self.data.update(data)

    @staticmethod
    def read_label_data(file_path, default_url=None):
        with open(file_path, 'r') as file:
            json_file = file.read()
            if not json_file:
                return {}
            data = json.loads(json_file)

        trans_data = {}
        for sample_name, sample_val in tqdm.tqdm(data.items()):
            sample_data = sample_val[0]
            sample_data['totalDesc'] = {
                'text_en':translate_zh2en(sample_data['totalDesc'], default_url),
                'text_zh':sample_data['totalDesc']
            }
            
            part_data = sample_data['partList'] if 'partList' in sample_data.keys() else []
            for data in part_data:
                data['desc'] = {
                    'text_en':translate_zh2en(data['desc'], default_url),
                    'text_zh':data['desc']
                }
            sample_data['partList'] = part_data

            frame_data = sample_data['frameList'] if 'frameList' in sample_data.keys() else []
            for data in frame_data:
                data['desc'] = {
                    'text_en':translate_zh2en(data['desc'], default_url),
                    'text_zh':data['desc']
                }
            sample_data['frameList'] = frame_data

            trans_data[sample_name] = sample_data
        return trans_data

    def get_instruction(self, uid, frames, text_en_augment=''):
        if uid.startswith("100"):
            # 对应包含的robot id的uid
            uid = "_".join(uid.split("_")[1:])
        if uid not in self.data.keys():
            return {
                'sub': [],
                'drawing': [],
                'text_zh': [],
                'text_en': [],
                'text_en_augment': [],
                'status': []
            }

        drawing = [LabelData.find_frame_by_time(frame_idx, self.data[uid]['frameList']) for frame_idx in frames]
        drawing = [element for sublist in drawing for element in sublist] # flatten to 1 d ['desc':{}, 'point':[], 'arrow':{}]

        return {
            'sub': [
                LabelData.find_description_by_time(frame_idx, self.data[uid]['partList']) for frame_idx in frames
            ],
            'drawing': drawing,
            'text_zh': [self.data[uid]['totalDesc']['text_zh']],
            'text_en': [self.data[uid]['totalDesc']['text_en']],
            'text_en_augment': [text_en_augment],
            'status': [
                LabelData.get_status(frame_idx, self.data[uid]['partList']) for frame_idx in frames
            ] # 0 normal, -1: failure, -2 : discard
        }
    
    @staticmethod
    def find_description_by_time(time, data):
        for item in data:
            if item['startTime'] <= time < item['endTime']:
                return {'text_zh': item['desc']['text_zh'], 'text_en': item['desc']['text_en']}
        return {'text_zh':'', 'text_en':''}

    @staticmethod
    def find_frame_by_time(time, data):
        ret = []
        for item in data:
            if item['time'] == time:
                ret.append({
                    'desc':{'text_zh': item['desc']['text_zh'], 'text_en': item['desc']['text_en']}, \
                    'point':[val for _,val in item['drawData'].items()] \
                                if 'start' not in item['drawData'].keys() \
                                else [],\
                    'arrow':{
                        key: [
                                val_ for _,val_ in val.items()
                             ]
                             for key, val in item['drawData'].items() # {'start':{'x':, 'y':}, 'end':{'x':, 'y':}}
                        } \
                            if 'start' in item['drawData'].keys() \
                            else {'start':[], 'end':[]}
                    }
                )
                # print_rank0('ret', ret)
        return ret

    @staticmethod
    def get_status(time, data):
        for item in data:
            if item['startTime'] <= time < item['endTime']:
                return item['status']
        return 0

from typing import List, Tuple
def compress_mask_coordinates(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    压缩mask数据，只存储mask=1的坐标。
    
    :param mask: 形状为 [n, h, w] 的numpy数组，其中n是帧数，h和w是高度和宽度
    :return: 
        1.压缩后的一维mask列表，由所有帧的mask坐标首尾连接组成 (shape为[num,], num为一个视频所有mask坐标个数)
        2.indexes列表，记录每个帧的全局起始索引和结束索引，用于恢复mask
    """
    start_indexes = []
    compressed_masks = []
    cur_start = 0
    for frame in mask:
        coords = np.argwhere(frame == 1).tolist()
        start_indexes.append([cur_start, cur_start + len(coords) - 1])
        cur_start = cur_start + len(coords)
        compressed_masks.extend(coords)

    return np.array(compressed_masks), np.array(start_indexes)

def restore_mask_coordinates(compressed_mask: np.ndarray, start_indexes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    从压缩数据还原mask。
    
    :param compressed_mask: 压缩后的mask列表，由所有帧的mask坐标首尾连接组成
    :param start_indexes: 每个帧的全局起始索引和结束索引
    :return: 还原后的mask，形状为 [len(start_indexes), h, w] 的numpy数组
    """
    h, w = shape
    # assert len(start_indexes) == n, "start_indexes.shape[0]!= n"
    restored_mask = np.zeros((len(start_indexes),h,w), dtype=bool)

    for i in range(len(start_indexes)):
        start, end = start_indexes[i]
        if end < start:
            continue
        coords = compressed_mask[start:end+1]
        for y, x in coords:
            restored_mask[i, y, x] = True
    return restored_mask

def get_random_idx(seq):
    if len(seq) == 1:
        return 0
    index = random.randint(0, len(seq) - 1)
    return index
class MaskMetaData:
    def __init__(self, file_path, mask_keys=['mask'], lang_keys=['instruction_en'], mask_shape=(480,640), lang_embed_file_path=None):
        with open(file_path, 'r') as file:
            json_file = file.read()
            if not json_file:
                return {}
            data = json.loads(json_file)
        self.file_path = file_path
        self.mask_keys = mask_keys
        self.lang_keys = lang_keys
        self.mask_shape = mask_shape
        self.default_mask = np.zeros(mask_shape, dtype=np.uint8)
        self.default_box = np.zeros(shape=(8,), dtype=np.uint8)
        # memory oom
        self.mask_frame_index = {} # used to index mask
        self.mask_meta_data = {} # list[dict], every dict is meta_data include instruction/face_mask/left_mask/right_mask and so on.
        self.lang_embed_file_path = lang_embed_file_path
        
        self.instruction_only = self.mask_keys is None or len(self.mask_keys) == 0 # 只用instruction
        self.instruction_embed = self.lang_embed_file_path is not None and os.path.exists(self.lang_embed_file_path) # 是否使用embeded instruction
        print_rank0(f'Instruction only: {self.instruction_only}, Instruction embed: {self.instruction_embed}')
        if self.instruction_embed:
            self.lang_embed_data = torch.load(self.lang_embed_file_path)
            print_rank0(f'load lang_embed_data from {self.lang_embed_file_path}')
            # get empty lang_embed_data, get embedding size from first sample
            lang_embed_size = self.lang_embed_data[list(self.lang_embed_data.keys())[0]]['embeddings'][0].shape[1]
            self.empty_lang_embed = torch.zeros((1,lang_embed_size))


        if self.instruction_only:
            for uid,value_list in data.items():
                if uid not in self.mask_meta_data:
                    self.mask_meta_data[uid] = {key:[] for key in self.lang_keys}
                for value in value_list['list']:
                    for key in self.lang_keys:
                        self.mask_meta_data[uid][key].append(value[key]) 
            # print self.mask_meta_data.len()
            print_rank0(f'Instruction only len is: {len(self.mask_meta_data)}')
        else:    
            for uid,value_list in data.items():
                if uid not in self.mask_meta_data:
                    self.mask_meta_data[uid] = {key:[] for key in self.mask_keys+self.lang_keys}
                if uid not in self.mask_frame_index:
                    self.mask_frame_index[uid] = {key:[] for key in self.mask_keys}
                for value in value_list['list']:
                    if value['face_quality'] == 0: # 过滤掉质量不好的数据
                        continue
                    for key in self.mask_keys:
                        mask_file_paths = value[key]
                        index_file_paths = [mask_file_path.split('.npy')[0]+'_indexes.npy' for mask_file_path in mask_file_paths] 
                        self.mask_meta_data[uid][key].append(mask_file_paths)
                        self.mask_frame_index[uid][key].append(index_file_paths)
                    for key in self.lang_keys:
                        self.mask_meta_data[uid][key].append(value[key])
            
                for key in self.mask_keys:
                    assert len(self.mask_meta_data[uid][key]) == len(self.mask_frame_index[uid][key]), f'In uid:{uid}, mask_key:{key}, len(self.mask_meta_data)={len(self.mask_meta_data[uid][key])} not equal len(self.mask_frame_index)={len(self.mask_frame_index[uid][key])}'
                self.first_mask_key = self.mask_keys[0]
                self.valid_uid = list(self.mask_meta_data.keys())[0]
                self.num_mask_type = len(self.mask_frame_index[self.valid_uid][self.first_mask_key][0]) # num of mask type: like static object and dynamic object mask 
                # print_rank0(f'self.num_mask_type:{self.num_mask_type}')
    
    def __dict__(self):
        return {'mask_meta_file_path': self.file_path}

    def get_instruction(self, uid, frames, lang_key='instruction_en'):
        # print_rank0(f'random_idx:{random_idx} of {len(self.mask_meta_data[uid][lang_key])}')
        instructions = {'text_en': ['' for _ in frames]}
        if uid in self.mask_meta_data:
            # random_idx = get_random_idx(self.mask_meta_data[uid][lang_key]) # 会导致与mask的random_idx不一致
            random_idx = 0
            instructions = {'text_en': [self.mask_meta_data[uid][lang_key][random_idx] for _ in frames]}
            if self.instruction_embed:
                if uid not in self.lang_embed_data:
                    print_rank0(f'uid:{uid} not in lang_embed_data', flush=True)
                instructions['lang_embed'] = [self.lang_embed_data[uid]['embeddings'][random_idx] for _ in frames]
        else:
            instructions = {'text_en': ['' for _ in frames]}
            if self.instruction_embed:
                instructions['lang_embed'] = [self.empty_lang_embed for _ in frames]
                print_rank0(f'uid:{uid} not in mask_meta_data', flush=True)
        return instructions
    
    def get_all_mask(self, uid, frame_idxs):
        all_masks = {}
        for mask_key in self.mask_keys:
            all_masks[mask_key] = self.get_mask(uid, frame_idxs, mask_key)
        return all_masks
        
    def get_compressed_mask(self, uid, frame_idxs, mask_key='face_mask'):
        h,w = self.mask_shape
        all_mask = [np.zeros((len(frame_idxs),h,w)) for _ in range(self.num_mask_type)]
        if uid in self.mask_meta_data:
            # random_idx = get_random_idx(self.mask_meta_data[uid][mask_key]) # 会导致与instruction的random_idx不一致
            random_idx = 0
            if len(self.mask_meta_data[uid][mask_key][random_idx]) > 0:
                mask_datas = [np.load(datapath) for datapath in self.mask_meta_data[uid][mask_key][random_idx] if os.path.exists(datapath)] # shape: (num_mask_type, num_mask)
                mask_indexes = [np.load(datapath) for datapath in self.mask_frame_index[uid][mask_key][random_idx] if os.path.exists(datapath)] # shape: (num_frames)
                if len(mask_datas) > 0 and len(mask_indexes) > 0:
                    all_mask = []
                    for mask_type_idx,mask_data in enumerate(mask_datas):
                        all_indexes = []
                        mask_type_indexes = mask_indexes[mask_type_idx]
                        for frame_idx in frame_idxs:
                            if frame_idx < len(mask_type_indexes): # not padding data
                                mask_index = mask_type_indexes[frame_idx]
                                all_indexes.append(mask_index)
                            else:
                                all_indexes.append(mask_type_indexes[-1]) # use last frame as padding
                        # print(f'all_indexes: {all_indexes[0].shape}, {len(all_indexes)}')
                        all_indexes = np.array(all_indexes)
                        raw_mask = restore_mask_coordinates(compressed_mask=mask_data, start_indexes=all_indexes, shape=(h,w))
                        all_mask.append(raw_mask)
            # print(f'mask shape when uid not exist: {len(all_mask)}, {all_mask[0].shape}')
        return all_mask 


    def get_mask(self, uid, frame_idxs, mask_key='mask'):
        all_mask1 = [self.default_mask for _ in frame_idxs]
        all_mask2 = [self.default_mask for _ in frame_idxs]
        if uid in self.mask_meta_data:
            meta_datas = self.mask_meta_data[uid][mask_key]
            if len(meta_datas) == 2:
                mask1 = np.load(meta_datas[0])
                mask2 = np.load(meta_datas[1])
                assert mask1.shape == mask2.shape, f'In {mask_key}: mask1 shape: {mask1.shape} != mask2 shape: {mask2.shape}'
                all_mask1 = []
                all_mask2 = []
                for idx in frame_idxs:
                    if idx < len(mask1):
                        all_mask1.append(mask1[idx])
                        all_mask2.append(mask2[idx])
                    else:
                        all_mask1.append(self.default_mask)
                        all_mask2.append(self.default_mask)
        return [all_mask1, all_mask2]

    def get_box(self, uid, frame_idxs):
        box = [self.default_box for _ in frame_idxs]
        if uid in self.box_meta_data:
            meta_datas = self.box_meta_data[uid]
            if len(meta_datas) == 2:
                raw_box = np.concatenate([np.load(meta_datas[0]),np.load(meta_datas[1])], axis=-1)
                box = []
                for idx in frame_idxs:
                    if idx < len(raw_box):
                        box.append(raw_box[idx])
                    else:
                        box.append(self.default_box)
        return box
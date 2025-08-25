import os
import json
import yaml
import time
import torch
import queue
import random
import hashlib
import threading
import traceback
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from collections import deque, abc
from torch.distributed import all_gather_object
from torchcodec.decoders import VideoDecoder
from x2robot_dataset.common.utils import balanced_split_between_processes, print_rank_0, decode_tensor
from x2robot_dataset.common.data_preprocessing import _ACTION_KEY_FULL_MAPPING, _ACTION_KEY_FULL_MAPPING_INV, process_action, is_stationary
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.instruction_processor import InstructionProcessor
from x2robot_dataset.common.data_utils import (
    convert_euler_to_6D,
    absolute_pose_to_relative_pose,
    relative_pose_to_absolute_pose,
    actions_to_relative,
    relative_to_actions,
)

mp.set_start_method("spawn", force=True)

def _default_check_episode_fn(
    topic_item, data_config: X2RDataProcessingConfig, data_chunk_config: X2RDataChunkConfig, check_mode: bool = True
):
    """
    此函数用于检查指定主题文件夹路径下的所有子文件夹，识别其中包含 JSON 文件的子文件夹，
    并从这些 JSON 文件中读取总帧数，最终返回包含子文件夹路径和有效帧数的字典列表。
    每个字典包含'path'和'length'键值对，以及原始topic_item中的所有其他键值对。

    参数:
    topic_item (dict): 包含主题信息的字典，其中'path'键对应要检查的主题文件夹路径。
    data_chunk_config (object): 数据块配置对象，该对象包含用于计算视频文件有效帧数的属性。
    check_mode (bool): 是否开启检查模式，默认开启。在检查模式下，会验证所有摄像头视频的帧数是否与JSON中记录的一致。
    支持接管数据：如果topic_item中有use_takeover_only=True，则只处理file_list中的接管段。
    支持降采样：如果topic_item中明确设置了sample_rate且< 1.0，则对数据进行降采样。
    
    返回:
    list: 包含处理结果的字典列表，若命中缓存则直接返回缓存内容，否则计算后返回并缓存结果。
    """
    # 设置缓存目录
    cache_dir = data_config.cache_dir

    # 创建缓存目录（如果不存在）
    os.makedirs(cache_dir, exist_ok=True)

    # 生成缓存键，添加接管模式和降采样模式到缓存键中
    use_takeover_only = topic_item.get("use_takeover_only", False)
    sample_rate = topic_item.get("sample_rate", None)
    folder_mtime = os.path.getmtime(topic_item["path"]) # 检测郑伟上一次更改文件时间，如果郑伟重跑解包，自动重新生成cache
    cache_key_parts = [
        topic_item["path"],
        str(folder_mtime),
        str(data_chunk_config.action_horizon),
        str(data_chunk_config.action_history_length),
        str(data_chunk_config.image_horizon),
        str(data_chunk_config.image_history_length),
        str(use_takeover_only),  # 添加接管模式标识
        str(sample_rate),  # 添加采样率标识
    ]
    cache_key = hashlib.sha256("".join(cache_key_parts).encode("utf-8")).hexdigest()

    # 缓存文件路径
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # 检查缓存是否存在
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            print(f"命中缓存: {cache_file}")
            return cached_data
        except Exception as e:
            print(f"缓存读取失败: {e}，将重新计算")

    # 缓存未命中，执行正常处理流程
    topic_folder_path = topic_item["path"]
    cam_mapping = data_config.cam_mapping
    result_list = []

    required_cameras = set(cam_mapping.keys())  # 获取所有需要检查的摄像头名称
    device = "cpu"  # 视频解码器设备，可以是"cpu"或"cuda"

    for root, dirs, files in os.walk(topic_folder_path):
        # 跳过空目录
        if not dirs and not files:
            continue

        # 使用集合提升查找效率
        invalid_files, phase_switch = set(), []
        
        invalid_keys = ['vague_sample', 'out_of_3_sigma', 'gripper_fast_oscillations', 
                    'select_low_quality', "select_not_standard"]

        # 预处理报告文件
        report_file = os.path.join(topic_folder_path, "report.json")
        if report_file and os.path.exists(report_file):  # 添加文件存在性检查
            with open(report_file, 'r') as f:
                report = json.load(f)
            for key in report:
                if key in invalid_keys:
                    invalid_files.update(report[key])  # 直接转为集合

        for sub_dir in dirs:

            base_name = os.path.basename(sub_dir)
            if base_name in invalid_files:
                print(f"Path: {sub_dir} 跳过，因为 {sub_dir} 在report.json中过滤")
                continue

            sub_dir_path = os.path.join(root, sub_dir)
            json_file = f"{sub_dir}.json"
            json_path = os.path.join(sub_dir_path, json_file)

            # 检查子目录是否存在且json文件存在
            if not os.path.isdir(sub_dir_path) or not os.path.isfile(json_path):
                continue

            # 获取子目录下所有MP4文件
            mp4_files = {f[:-4] for f in os.listdir(sub_dir_path) if f.endswith(".mp4")}

            # 检查是否存在所有需要的摄像头视频
            if not required_cameras.issubset(mp4_files):
                missing_cams = required_cameras - mp4_files
                print(f"Path: {sub_dir_path} 缺少视频文件: {', '.join(missing_cams)}")
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    total_frames = data.get("total", 0)
                    file_list = data.get("file_list", {})

                    # 检查模式：验证所有摄像头视频的帧数是否与JSON中记录的一致
                    if check_mode:
                        # print(f"开始检查目录 {sub_dir_path} 中视频文件的帧数...")
                        # check_start_time = time.time()
                        frame_check_passed = True

                        for cam in required_cameras:
                            video_path = os.path.join(sub_dir_path, f"{cam}.mp4")
                            try:
                                # 使用torchcodec读取视频元数据
                                decoder = VideoDecoder(video_path, device=device)
                                actual_frames = decoder.metadata.num_frames

                                if actual_frames != total_frames:
                                    print(f"Path: {sub_dir_path} 警告: 摄像头 {cam} 的帧数({actual_frames})与JSON记录的总帧数({total_frames})不一致，跳过此目录")
                                    frame_check_passed = False
                                    break  # 一旦发现不一致，立即跳出循环

                                # 释放解码器资源
                                del decoder
                            except Exception as e:
                                print(f"Path: {sub_dir_path} 错误: 无法读取摄像头 {cam} 的视频帧数: {e}")
                                frame_check_passed = False
                                break  # 发生错误时跳过此目录

                        # check_end_time = time.time()
                        # check_duration = check_end_time - check_start_time
                        # print(f"视频帧数检查完成，耗时 {check_duration:.2f} 秒")
                        
                        # 使用默认的统一映射检查动作数据
                        action_keys_raw2predict = { _ACTION_KEY_FULL_MAPPING[k]:k for k in data_config.predict_action_keys}
                        action_data = process_action(data, raw_key2model_key=action_keys_raw2predict, filter_angle_outliers=data_config.filter_angle_outliers)
                        
                        if not set(data_config.predict_action_keys).issubset(action_data.keys()):
                            missing_actions = set(data_config.predict_action_keys) - set(action_data.keys())
                            print(f"Path: {sub_dir_path} 错误: 警告: 动作数据缺少以下预测键: {', '.join(missing_actions)}，跳过此目录")
                            frame_check_passed = False
                        
                        # 检查观测动作键是否都存在
                        action_keys_raw2obs = { _ACTION_KEY_FULL_MAPPING[k]:k for k in data_config.obs_action_keys}
                        action_data_obs = process_action(data, raw_key2model_key=action_keys_raw2obs, filter_angle_outliers=data_config.filter_angle_outliers)
                        
                        if not set(data_config.obs_action_keys).issubset(action_data_obs.keys()):
                            missing_obs_actions = set(data_config.obs_action_keys) - set(action_data_obs.keys())
                            print(f"Path: {sub_dir_path} 错误: 警告: 动作数据缺少以下观测键: {', '.join(missing_obs_actions)}，跳过此目录")
                            frame_check_passed = False

                        # 如果检查失败，则跳过此目录
                        if not frame_check_passed:
                            continue

                    # 接管数据处理逻辑
                    if use_takeover_only:
                        # 检查是否有file_list且包含接管段
                        if not file_list:
                            print(f"Path: {sub_dir_path} 启用接管模式但没有找到file_list，跳过")
                            continue

                        # 提取接管段
                        takeover_segments = []
                        for key, value in file_list.items():
                            if "接管" in key:  # 识别接管段
                                try:
                                    start_frame, end_frame = map(int, value.split())
                                    takeover_segments.append((start_frame, end_frame, key))
                                except ValueError:
                                    print(f"Warning: 无法解析接管段 {key}: {value}")
                                    continue

                        if not takeover_segments:
                            print(f"Path: {sub_dir_path} 启用接管模式但没有找到接管段，跳过")
                            continue

                        # 为每个接管段创建episode项
                        for segment_idx, (start_frame, end_frame, segment_name) in enumerate(takeover_segments):
                            # 调整帧范围以适应数据块配置
                            if data_chunk_config.left_padding:
                                st_frame = start_frame + data_chunk_config.action_history_length
                            else:
                                st_frame = start_frame

                            if data_chunk_config.right_padding:
                                ed_frame = end_frame - data_chunk_config.action_horizon
                            else:
                                ed_frame = end_frame

                            # 确保有足够的帧
                            if ed_frame <= st_frame:
                                print(f"Path: {sub_dir_path} 接管段 {segment_name} 帧数不足，跳过")
                                continue

                            num_frames = ed_frame - st_frame

                            # 创建接管段的episode项，同时传递降采样参数
                            item_dict = {}
                            item_dict.update(topic_item)
                            item_dict.update({
                                "path": sub_dir_path,
                                "length": total_frames,
                                "num_frames": num_frames,
                                "st_frame": st_frame,
                                "ed_frame": ed_frame,
                                "takeover_segment": segment_name,
                                "original_start": start_frame,
                                "original_end": end_frame,
                                "segment_index": segment_idx,
                                "is_takeover_data": True,
                                "sample_rate": sample_rate,
                            })

                            result_list.append(item_dict)
                            print_rank_0(f"添加接管段: {sub_dir_path} {segment_name} ({start_frame}-{end_frame})")

                    else:
                        # 原有的正常数据处理逻辑
                        # 机械臂停止时不纳入训练样本
                        if data_config.trim_stationary: 
                            action_data = process_action(data, filter_angle_outliers=data_config.filter_angle_outliers)
                            threshold = 0.01
                            keys = action_data.keys()
                            start, end = 0, total_frames

                            while start < end - 1:
                                frame_data = [action_data[key][start : start + 2] for key in keys]
                                if not is_stationary(frame_data, threshold):
                                    break
                                start += 1

                            while end > start + 1:
                                frame_data = [action_data[key][end - 2 : end] for key in keys]
                                if not is_stationary(frame_data, threshold):
                                    break
                                end -= 1

                            trim_st = start
                            trim_end = end
                        else:
                            trim_st = 0
                            trim_end = total_frames

                        # 计算起始帧和结束帧的索引（左闭右开）以及有效帧数
                        if data_chunk_config.left_padding:
                            st_frame = trim_st + data_chunk_config.action_history_length
                        else:
                            st_frame = trim_st

                        if data_chunk_config.right_padding:
                            ed_frame = trim_end - data_chunk_config.action_horizon
                        else:
                            ed_frame = trim_end

                        num_frames = ed_frame - st_frame

                        # 创建字典，包含path和length，以及其他在topic_item中的所有key_value
                        item_dict = {}
                        item_dict.update(topic_item)
                        item_dict.update(
                            {
                                "path": sub_dir_path,
                                "length": total_frames,
                                "num_frames": num_frames,
                                "st_frame": st_frame,
                                "ed_frame": ed_frame,
                                "is_takeover_data": False,
                                "sample_rate": sample_rate,
                            }
                        )
                        
                        result_list.append(item_dict)

            except Exception as e:
                print(f"Error processing {json_path}: {e}")

    # 保存结果到缓存
    try:
        with open(cache_file, "w") as f:
            json.dump(result_list, f)
        print(f"已将结果缓存至: {cache_file}")
    except Exception as e:
        print(f"缓存保存失败: {e}")

    return result_list


def _default_get_frame_fn(episode_item, data_config: X2RDataProcessingConfig, data_chunk_config: X2RDataChunkConfig, instruction_processor: InstructionProcessor = None, distributed_instruction_ratio: float = 1.0):
    """
    该函数用于处理给定 Episode 的数据，包括多视角视频数据、动作数据、触觉数据以及关节电流/力矩数据，
    并根据配置信息将这些数据组织成帧的形式返回，支持当前帧 + 历史帧 + 动作预测窗口的数据切片。

    参数:
    episode_item (dict): 包含 Episode 相关信息的字典，至少应包含：
                         - "path": 数据所在文件夹路径；
                         - "st_frame": 起始帧索引；
                         - "ed_frame": 结束帧索引。
    data_config (X2RDataProcessingConfig): 数据处理配置对象，控制是否解析触觉数据、摄像头映射、低维观测长度等参数。
    data_chunk_config (X2RDataChunkConfig): 数据块配置对象，定义了动作窗口长度（action_horizon）、图像窗口长度（image_horizon）、是否合并历史动作（merge_cur_history）、是否合并历史图像（merge_image_history）、action_history_length、image_history_length等。

    返回:
    list: 包含帧数据的列表，每个元素是一个字典，包含以下键值对：
          - "action": 当前帧起始的动作序列，形状为 [action_horizon, action_dim]。
          - "obs": 观测数据字典，包含 agent_pos 和多个摄像头名称对应的视频帧序列, 如果merge_cur_history为False且action_history_length>0，则包含action_histories, 如果merge_image_history为False且image_history_length>0，则包含camera_histories。
          - "uid": 样本的唯一标识符，由 robot_id 和 sample_name 组成，并右填充至固定长度 200。
          - "frame": 当前帧的索引。
          - "dataset_name": 数据集名称，由 task_id 和 sample_name 组成，并右填充至固定长度 100。

    注意:
    - 目前不支持 filter_angle_outliers=True 的角度异常过滤。
    - 所有张量默认使用 torch.float32 类型。
    - 使用 VideoDecoder 加载视频帧，支持 GPU 或 CPU 解码。
    """
    # 设置和读取参数
    obs_data = {}
    device = "cpu"

    folder_path = episode_item["path"]
    filter_angle_outliers = data_config.filter_angle_outliers
    low_dim_obs_horizon = data_config.low_dim_obs_horizon
    cam_mapping = data_config.cam_mapping

    assert filter_angle_outliers is False, "filter_angle_outliers must be False for now"

    # 处理多视角视频数据
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filename = file.split(".")[0]
            if file.endswith(".mp4") and filename in data_config.cam_mapping.keys():
                try:
                    video_path = os.path.join(root, file)
                    decoder = VideoDecoder(video_path, device=device)
                    obs_data[filename] = decoder
                except Exception as e:
                    print(f"Error opening video file {file}: {e}")

    # 构造predict_action_keys和obs_action_keys
    assert data_config.predict_action_keys is not None, "predict_action_keys must be configured in data_config"
    assert data_config.obs_action_keys is not None, "obs_action_keys must be configured in data_config"
    action_keys_raw2predict = { _ACTION_KEY_FULL_MAPPING[k]:k for k in data_config.predict_action_keys}
    action_keys_raw2obs = { _ACTION_KEY_FULL_MAPPING[k]:k for k in data_config.obs_action_keys}
    
    # process_action会将原始数据key映射为模型key
    action_data = process_action(folder_path, raw_key2model_key=action_keys_raw2predict, filter_angle_outliers=filter_angle_outliers) 
    action_data_obs = process_action(folder_path, raw_key2model_key=action_keys_raw2obs,filter_angle_outliers=filter_angle_outliers) 
    
    # 检查是否启用降采样
    sample_rate = episode_item.get("sample_rate", None)
    
    # 生成原始帧索引范围
    original_start_frame = episode_item["st_frame"]
    original_end_frame = episode_item["ed_frame"]
    total_frames = original_end_frame - original_start_frame
    
    # 根据sample_rate决定是否启用降采样
    if sample_rate is not None and sample_rate < 1.0:
        # 生成采样索引
        sample_indices = np.linspace(0, total_frames - 1, 
                                   int(total_frames * sample_rate),
                                   dtype=int)
        # 将相对索引转换为绝对索引
        frame_indices = [original_start_frame + idx for idx in sample_indices]
        print_rank_0(f"启用降采样: 原始帧数{total_frames} -> 采样帧数{len(frame_indices)}, 采样率{sample_rate}")
    else:
        # 不降采样，使用原始帧索引
        frame_indices = list(range(original_start_frame, original_end_frame))
    
    frames = []

    for frame_idx in frame_indices:
        frame = {}

        # 设置样本的动作数据
        action_st = frame_idx
        # Define: H (Predict Action Horizon) + L (Past Obs Horizon)
        # Define: o (observation)
        # o_(t-L),..., o_t, o_(t+1),..., o_(t+H)
        action_ed = frame_idx + data_chunk_config.action_horizon + low_dim_obs_horizon  
        predict_action_slices = []
        obs_action_slices = []

        # # 处理预测动作 - 使用predict_action_keys从映射后的action_data中取数据
        for key in data_config.predict_action_keys:
            action_window = action_data[key][action_st:action_ed]
            if action_window.ndim == 1:
                action_window = action_window[:, np.newaxis]  # 变成 (N, 1)

            # convert Euler angles to 6D rotation matrix
            if 'rotation' in key and data_config.use_6D_rotation is True:
                action_window = convert_euler_to_6D(action_window)

            predict_action_slices.append(action_window)

        # 处理观察动作 - 使用obs_action_keys从映射后的action_data中取数据
        for key in data_config.obs_action_keys:
            # 此时取原始数据，不进行映射
            action_window = action_data_obs[key][action_st:action_ed]

            # 如果是一维数组，增加一个维度后再转置
            if action_window.ndim == 1:
                action_window = action_window[:, np.newaxis]  # 变成 (N, 1)

            # convert Euler angles to 6D rotation matrix
            if 'rotation' in key and data_config.use_6D_rotation is True:
                action_window = convert_euler_to_6D(action_window)
            
            obs_action_slices.append(action_window)

        
        # 处理action histories - 只要action_history_length > 0就生成
        if data_chunk_config.action_history_length > 0:
            frame["action_histories"] = {}
            action_history_st = max(action_st - data_chunk_config.action_history_length, 0)
            action_history_ed = action_st
            
            for key in data_config.obs_action_keys:
                action_history_window = action_data_obs[key][action_history_st:action_history_ed]
                
                if action_history_window.ndim == 1:
                    action_history_window = action_history_window[:, np.newaxis]

                # convert Euler angles to 6D rotation matrix
                if 'rotation' in key and data_config.use_6D_rotation is True:
                    action_history_window = convert_euler_to_6D(action_history_window)

                # Left padding if needed
                if data_chunk_config.left_padding and action_history_window.shape[0] < data_chunk_config.action_history_length:
                    pad_size = data_chunk_config.action_history_length - action_history_window.shape[0]
                    if action_history_window.shape[0] > 0:
                        padding_value = action_history_window[0]
                    else:
                        # 如果没有历史数据，用当前帧的第一个值填充
                        current_action = action_data_obs[key][action_st:action_st+1]
                        if current_action.ndim == 1:
                            current_action = current_action[:, np.newaxis]

                        # convert Euler angles to 6D rotation matrix
                        if 'rotation' in key and data_config.use_6D_rotation is True:
                            current_action = convert_euler_to_6D(current_action)

                        padding_value = current_action[0]
                    
                    padding = np.broadcast_to(padding_value, (pad_size, *action_history_window.shape[1:]))
                    action_history_window = np.concatenate([padding, action_history_window], axis=0)
                
                frame["action_histories"][key] = action_history_window
        
        # 拼接动作数据
        predict_action = torch.from_numpy(np.concatenate(predict_action_slices, axis=1)) if predict_action_slices else torch.empty(0)
        obs_action = torch.from_numpy(np.concatenate(obs_action_slices, axis=1)) if obs_action_slices else torch.empty(0)
        
        # 处理预测动作 - 根据use_relative_action决定是否转换为相对动作
        if data_chunk_config.use_relative_action == True:
            assert low_dim_obs_horizon == 1, f"only low_dim_obs_horizon=1 is being supported"
            # Convert action from Absolute EE to Relative EE
            # Special tips: the first frame should be agent pose
            # predict_action[0] = obs_action[0] # bug, if predict_action not same as obs_action, the first frame will be wrong
            predict_action_dim = predict_action.shape[-1]
            predict_action[0] = obs_action[0][:predict_action_dim] 
            predict_action = torch.from_numpy(absolute_pose_to_relative_pose(predict_action, data_config.predict_action_keys, data_config, data_chunk_config))
        
        
        frame["action"] = predict_action[low_dim_obs_horizon:, :]

        # 设置样本的观测数据（输入）
        obs_st = frame_idx
        obs_ed = frame_idx + data_chunk_config.image_horizon
        frame["obs"] = {}
        
        # 处理agent_pos - 根据merge_cur_history决定是否合并历史
        if (data_chunk_config.merge_cur_history and 
            data_chunk_config.action_history_length > 0):
            if data_chunk_config.use_relative_action == True:
                assert False, "use_relative_action=True with merge_cur_history=True is not supported for now: Need more experiments"
            # 合并历史动作到agent_pos
            history_action_slices = []
            for key in data_config.obs_action_keys:
                if key in frame["action_histories"]:
                    history_action_slices.append(frame["action_histories"][key])
            
            if history_action_slices:
                action_history = torch.from_numpy(np.concatenate(history_action_slices, axis=1))
                current_obs_action = obs_action[:low_dim_obs_horizon, :]
                frame["obs"]["agent_pos"] = torch.cat([action_history, current_obs_action], dim=0)
            else:
                frame["obs"]["agent_pos"] = obs_action[:low_dim_obs_horizon, :]
        else:
            frame["obs"]["agent_pos"] = obs_action[:low_dim_obs_horizon, :]
        
        # 处理observation histories - 只要image_history_length > 0就生成
        if data_chunk_config.image_history_length > 0:
            frame["camera_histories"] = {}
            obs_history_st = max(obs_st - data_chunk_config.image_history_length, 0)
            obs_history_ed = obs_st
            
            for key in obs_data.keys():
                if obs_history_ed > obs_history_st:
                    obs_history = obs_data[key][obs_history_st:obs_history_ed].clone().permute(0, 2, 3, 1)
                    
                    # Left padding if needed
                    if data_chunk_config.left_padding and obs_history.shape[0] < data_chunk_config.image_history_length:
                        pad_size = data_chunk_config.image_history_length - obs_history.shape[0]
                        if obs_history.shape[0] > 0:
                            padding_value = obs_history[0:1]
                        else:
                            # 如果没有历史数据，用当前帧的第一个值填充
                            current_obs = obs_data[key][obs_st:obs_st+1].clone().permute(0, 2, 3, 1)
                            padding_value = current_obs[0:1]
                        
                        padding = padding_value.repeat(pad_size, 1, 1, 1)
                        obs_history = torch.cat([padding, obs_history], dim=0)
                    
                    frame["camera_histories"][cam_mapping[key]] = obs_history
        
        # 处理摄像头观测 - 根据merge_image_history决定是否合并历史
        for key in obs_data.keys():
            current_obs = obs_data[key][obs_st:obs_ed].clone().permute(0, 2, 3, 1)
            
            if (data_chunk_config.merge_image_history and 
                data_chunk_config.image_history_length > 0):
                # 合并历史图像到当前观测
                history_obs = frame["camera_histories"][cam_mapping[key]]
                frame["obs"][cam_mapping[key]] = torch.cat([history_obs, current_obs], dim=0)
            else:
                frame["obs"][cam_mapping[key]] = current_obs

        # 使用传入的instruction_processor
        if instruction_processor is not None:
            frame["instruction"] = instruction_processor.get_frame_instruction_from_path(
                episode_item["path"], frame_idx
            )
        else:
            frame["instruction"] = data_config.default_instruction or ""

        if distributed_instruction_ratio > 0.0:
            # 处理instruction （下发式）
            diversity_path = os.path.join(folder_path, "diversity.json")
            try:
                with open(diversity_path, "r") as f:
                    diversity_data_dict = json.load(f)
            except Exception as e:
                print(f"Error opening diversity file {diversity_path}: {e}")
                diversity_data_dict = None

            # Select instruction from distribute based on frame_idx
            selected_instruction = diversity_data_dict.get("instruction", "")  # Default to main instruction

            # Check if frame_idx falls within any range in distribute
            distribute = diversity_data_dict.get("distribute", {})
            for range_key, instruction in distribute.items():
                try:
                    start_frame, end_frame = map(int, range_key.split())
                    # if start_frame <= frame_idx < end_frame:
                    # 改为 action_chunk 全在 range 内才使用
                    if start_frame <= obs_st and obs_ed <= end_frame and random.random() < distributed_instruction_ratio:
                        selected_instruction = instruction
                except (ValueError, KeyError):
                    # Skip invalid range keys
                    continue

            # Remove everything after '|' character e.g. "把芬达放到桌子上|无干扰" -> "把芬达放到桌子上"
            if '|' in selected_instruction:
                selected_instruction = selected_instruction.split('|')[0]
            if selected_instruction == "从木盒里拿起内胆":
                inside_color = diversity_data_dict['diversity']['内胆颜色']
                selected_instruction = f"要{inside_color}内胆，" + selected_instruction 
            frame["instruction"] = selected_instruction

        # 构造字符串标识
        robot_id, task_id, sample_name = [s for s in folder_path.split("/")[-3:] if s != ""]
        
        frame["uid"] = robot_id + "_" + sample_name
        frame["dataset_name"] = task_id + "/" + sample_name
        frame["frame"] = frame_idx
        frames.append(frame)
    return frames


def _collate_nested_dicts(batch):
    """
    将一个包含嵌套字典结构的样本列表 collate 成一个统一结构的 batch 字典，
    其中所有 Tensor 都被 stack。
    """
    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return torch.stack(batch)

    elif isinstance(elem, dict):
        result = {}
        for key in elem:
            result[key] = _collate_nested_dicts([d[key] for d in batch])
        return result

    elif isinstance(elem, abc.Mapping):
        return {key: _collate_nested_dicts([d[key] for d in batch]) for key in elem}

    elif isinstance(elem, (tuple, list)):
        return type(elem)(_collate_nested_dicts(samples) for samples in zip(*batch))

    elif isinstance(elem, str):
        # 对于字符串（如指令），返回字符串列表
        return batch

    elif isinstance(elem, int):
        # 对于字符串（如指令），返回字符串列表
        return batch

    else:
        # 非 Tensor、非 dict、非 list/tuple 类型原样返回（不参与 batch）
        return batch[0]


class DynamicRobotDataset:
    def __init__(
        self,
        dataset_config_path: str,
        data_config: X2RDataProcessingConfig,
        data_chunk_config: X2RDataChunkConfig,
        preload_size: int = 128,
        buffer_size: int = 20000,
        rank: int = -1,
        world_size: int = -1,
        batch_size: int = 32,
        check_episode_fn=_default_check_episode_fn,
        get_frame_fn=_default_get_frame_fn
    ):
        """
        初始化 DynamicRobotDataset 类的实例。

        参数:
        dataset_config_path (str): 包含了所有输入数据集配置的配置文件路径
        data_configs (List[X2RDataProcessingConfig]): 每个Topic对象的配置列表。
        data_chunk_config (X2RDataChunkConfig): 数据划分配置对象。
        buffer_size (int, 可选): 数据缓冲区的大小，默认为 2000。
        rank (int, 可选): 当前进程的排名，默认为 -1。
        world_size (int, 可选): 分布式训练中的进程总数，默认为 -1。
        batch_size (int, 可选): 每个批次的数据样本数量，默认为 16。
        check_episode_fn (function, 可选): 用于检查 episode 信息的函数，默认为 None。
        get_frame_fn (function, 可选): 用于获取帧数据的函数，默认为 None。
        collate_fn (function, 可选): 用于整理数据样本的函数，默认为 None。
        """
        self.rank = rank
        self.check_episode_fn = check_episode_fn
        self.get_frame_fn = get_frame_fn
        self.distributed_instruction_ratio = data_config.get("distributed_instruction_ratio", 0.0)
        self.ctx = mp.get_context("spawn")  # 创建spawn上下文
        self.train_batch_queue = self.ctx.Queue(maxsize=preload_size)
        self.val_batch_queue = self.ctx.Queue(maxsize=preload_size)
        self._stop_event = mp.Event()  # 使用 multiprocessing.Event 来进行进程间通信

        # 计算全局的batch数，并使用共享变量存储
        self.global_train_iters = mp.Value("i", -1)
        self.global_val_iters = mp.Value("i", -1)

        # 创建instruction处理器实例
        self.instruction_processor = InstructionProcessor(data_config)

        # 启动新进程
        self.process = mp.Process(
            target=self._run_threads,
            args=(
                dataset_config_path,
                data_config,
                data_chunk_config,
                world_size,
                rank,
                buffer_size,
                batch_size,
                self._stop_event,
                self.global_train_iters,
                self.global_val_iters,
            ),
        )
        self.process.start()

        while self.global_train_iters.value == -1 or self.global_val_iters.value == -1:
            time.sleep(1)
            print_rank_0(f"[_init] 等待全局迭代器初始化完成...")

    def parse_dataset(self, yaml_file_path):
        try:
            with open(yaml_file_path, "r") as file:
                data = yaml.safe_load(file)
                dataset_paths = data.get("dataset_path", [])
                return dataset_paths
        except FileNotFoundError:
            print(f"错误: 文件 {yaml_file_path} 未找到。")
        except yaml.YAMLError as e:
            print(f"错误: 解析YAML文件时出错: {e}")
        return []

    def _run_threads(
        self,
        dataset_config_path,
        data_config,
        data_chunk_config,
        world_size,
        rank,
        buffer_size,
        batch_size,
        stop_event,
        global_train_iters,
        global_val_iters,
    ):
        dist.init_process_group(
            backend="gloo",
            store=dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"], port=int(os.environ["MASTER_PORT"]) + 1, world_size=world_size, is_master=(rank == 0)
            ),
            world_size=world_size,
            rank=rank,
        )
        """启动所有工作线程"""
        train_episodes = []
        val_episodes = []
        threads = [
            threading.Thread(
                target=self._monitor_config,
                args=(
                    train_episodes,
                    val_episodes,
                    global_train_iters,
                    global_val_iters,
                    dataset_config_path,
                    data_config,
                    data_chunk_config,
                    world_size,
                    rank,
                    batch_size,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=self._process_data_v2,
                args=(train_episodes, self.train_batch_queue, buffer_size, batch_size, stop_event, data_config, data_chunk_config),
                daemon=True,
            ),
            threading.Thread(
                target=self._process_data_v2,
                args=(val_episodes, self.val_batch_queue, buffer_size, batch_size, stop_event, data_config, data_chunk_config),
                daemon=True,
            ),
        ]

        for t in threads:
            t.start()

        # 主线程保持存活
        while any(t.is_alive() for t in threads):
            time.sleep(1)

    def _monitor_config(
        self,
        train_episodes,
        val_episodes,
        global_train_iters,
        global_val_iters,
        dataset_config_path,
        data_config,
        data_chunk_config,
        world_size,
        rank,
        batch_size,
    ):
        dataset_items = []
        all_train_samples = [0] * world_size
        all_val_samples = [0] * world_size
        last_modified_time = 0
        # 将文件不变化的等待时间设置为单独的变量
        wait_time = 30
        while True:
            try:
                current_modified_time = os.path.getmtime(dataset_config_path)
                if current_modified_time > last_modified_time:

                    print_rank_0(f"[Info - Dataset] 检测到配置文件变化，等待{wait_time}秒后，将检查配置是否可以解析。")
                    if last_modified_time != 0:
                        time.sleep(wait_time)  # 等待wait_time秒确保文件不再变化
                    new_modified_time = os.path.getmtime(dataset_config_path)
                    if new_modified_time == current_modified_time:
                        # 解析新增的Topic的episodes
                        new_dataset_items = self.parse_dataset(dataset_config_path)
                        if new_dataset_items != dataset_items:
                            print_rank_0(f"[Info - Dataset] 配置可以解析，将进行数据集更新！")

                            # 根据新增的 path 找出对应的新 topic 字典
                            new_paths = {item["path"] for item in new_dataset_items}
                            old_paths = {item["path"] for item in dataset_items}
                            new_paths_diff = new_paths - old_paths
                            new_topics = [item for item in new_dataset_items if item["path"] in new_paths_diff]

                            # 平均分配episode到各个进程进行处理
                            episode_frames = []
                            with balanced_split_between_processes(
                                inputs=new_topics, num_processes=world_size, process_index=rank, apply_padding=False
                            ) as topic_items:
                                for idx, topic_item in enumerate(topic_items):
                                    print(f"[rank {rank} - ({idx+1}/{len(topic_items)})] Processing {topic_item}", flush=True)
                                    episode_frames.extend(self.check_episode_fn(topic_item, data_config, data_chunk_config))
                                all_episode_frames = [None] * world_size
                                all_gather_object(all_episode_frames, episode_frames)
                                episode_frames = [item for sublist in all_episode_frames if sublist is not None for item in sublist]

                            sorted_episodes = sorted(episode_frames, key=lambda x: x["num_frames"], reverse=True)

                            # 按照贪心算法分配episode
                            for episode in sorted_episodes:
                                frames = episode["num_frames"]
                                min_train_samples = min(all_train_samples)
                                min_val_samples = min(all_val_samples)
                                total_samples = min_train_samples + min_val_samples
                                current_train_ratio = min_train_samples / total_samples if total_samples > 0 else 0
                                target_train_ratio = data_config.train_test_split

                                if current_train_ratio < target_train_ratio:
                                    # 分配到 train
                                    min_rank = all_train_samples.index(min_train_samples)
                                    all_train_samples[min_rank] += frames
                                    if min_rank == rank:
                                        train_episodes.append(episode)
                                else:
                                    # 分配到 val
                                    min_rank = all_val_samples.index(min_val_samples)
                                    all_val_samples[min_rank] += frames
                                    if min_rank == rank:
                                        val_episodes.append(episode)

                            # 更新全局最少的samples数
                            global_train_samples = min(all_train_samples)
                            global_val_samples = min(all_val_samples)

                            # 更新全局的batch数
                            global_train_iters.value = global_train_samples // batch_size
                            global_val_iters.value = global_val_samples // batch_size

                            dataset_items = new_dataset_items

                    last_modified_time = current_modified_time
            except Exception as e:
                print_rank_0(f"检测到配置文件变化，但更新时发生错误，请检查数据文件的格式是否正确等。错误信息： {e}")
                error_stack = traceback.format_exc()
                print_rank_0(f"错误栈信息：\n{error_stack}")
            time.sleep(1)

    def _process_data_v2(self, original_episodes, batch_queue, buffer_size, batch_size, stop_event, data_config, data_chunk_config):
        """二合一的通用数据处理线程 (循环处理模式)"""

        while len(original_episodes) == 0:
            time.sleep(1)
            # print_rank_0(f"[_process_data] 等待数据集加载完成...")

        buffer = deque(maxlen=int(buffer_size * 1.5))

        while not stop_event.is_set():
            # 创建本轮次快照
            episode_snapshot = original_episodes.copy()
            random.shuffle(episode_snapshot)

            for episode in episode_snapshot:
                while len(buffer) >= buffer_size and not stop_event.is_set():
                    indices = sorted(random.sample(range(len(buffer)), k=batch_size), reverse=True)
                    batch = [buffer[i] for i in indices]
                    # 删除已使用的样本
                    for i in indices:
                        del buffer[i]
                
                    try:
                        # 尝试执行数据拼接
                        batched = _collate_nested_dicts(batch)
                    except Exception as e:
                        # 打印错误信息和有问题的样本
                        print(f"Error in _collate_nested_dicts: {e}")
                        print("Batched samples causing error:")
                        for i, sample in enumerate(batch):
                            try:
                                uid = sample.get("uid", "unknown")
                                dataset_name = sample.get("dataset_name", "unknown")
                                print(f"Sample {i}: uid={decode_tensor(uid)}, dataset_name={decode_tensor(dataset_name)}")
                            except Exception as inner_e:
                                print(f"Error accessing metadata for sample {i}: {inner_e}")
                        exit()
                    while not stop_event.is_set():
                        try:
                            batch_queue.put(batched, block=True, timeout=0.01)
                            break
                        except queue.Full:
                            continue
                # 打印时间戳
                # print(f"[current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [Rank {self.rank}] 开始处理数据，当前缓冲区大小: {len(buffer)}, 当前BatchQueue大小: {batch_queue.qsize()}")
                frames = self.get_frame_fn(episode, data_config, data_chunk_config, self.instruction_processor, self.distributed_instruction_ratio)
                # print(f"[current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [Rank {self.rank}] 处理数据完毕，获得帧数: {len(frames)}, 当前BatchQueue大小: {batch_queue.qsize()}")
                buffer.extend(frames)

    def get_train_dataloader(self):
        while self.global_train_iters.value == -1:
            print_rank_0(f"[Rank {self.rank}] 等待全局训练迭代次数初始化...")
            time.sleep(1)
        return DoubleBufferedRobotDataLoader(data_buffer=self.train_batch_queue, max_iters=self.global_train_iters.value)

    def get_val_dataloader(self):
        while self.global_train_iters.value == -1:
            print_rank_0(f"[Rank {self.rank}] 等待全局训练迭代次数初始化...")
            time.sleep(1)
        return DoubleBufferedRobotDataLoader(data_buffer=self.val_batch_queue, max_iters=self.global_val_iters.value)

    def __del__(self):
        """析构函数，停止所有线程和进程并释放资源"""
        print(f"[Rank {self.rank}] Dataset开始清理资源")
        # 设置停止事件
        self._stop_event.set()

        # 终止进程
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


class DoubleBufferedRobotDataLoader:
    def __init__(self, data_buffer: mp.Queue, max_iters: int, device=None):
        """
        使用队列实现的双缓冲 DataLoader。

        参数:
        data_buffer (mp.Queue): 原始数据队列。
        max_iters (int): 最大迭代次数。
        device (str or None): 目标设备，默认根据 rank 自动选择。
        """
        self.data_buffer = data_buffer
        self.max_iters = max_iters
        self.iter_count = 0

        # 自动选择设备
        if device is None:
            if dist.is_initialized():
                rank = dist.get_rank()
                self.device = f"cuda:{rank % torch.cuda.device_count()}"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # 双缓冲队列，最多缓存1个batch
        self.buffer_queue = queue.Queue(maxsize=1)

        # 启动后台预加载线程
        self.stop_event = threading.Event()
        self.loader_thread = threading.Thread(target=self._preload, daemon=True)
        self.loader_thread.start()

    def _preload(self):
        """后台线程：持续预加载并搬运数据到目标设备，达到 max_iters 后自动退出"""
        prefetched_count = 0  # 新增计数器

        while not self.stop_event.is_set() and prefetched_count < self.max_iters:
            # Step 1: 从原始队列获取数据
            while not self.stop_event.is_set():
                try:
                    batch = self.data_buffer.get(block=True, timeout=0.1)
                    break
                except queue.Empty:
                    continue

            # Step 2: 移动到目标设备
            cuda_batch = move_to_cuda(batch, self.device)

            # Step 3: 放入缓冲队列（支持阻塞等待空间）
            while not self.stop_event.is_set():
                try:
                    self.buffer_queue.put(cuda_batch, block=True, timeout=0.1)
                    prefetched_count += 1  # 成功放入后计数 +1
                    break
                except queue.Full:
                    continue

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count >= self.max_iters:
            self.stop_event.set()
            raise StopIteration

        batch = self.buffer_queue.get(block=True)
        self.iter_count += 1
        return batch

    def __len__(self):
        return self.max_iters

    def __del__(self):
        self.stop_event.set()


def move_to_cuda(obj, device="cuda"):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    else:
        return obj

if __name__ == "__main__":
    episode_item = {
        "path": '/x2robot_v2/xinyuanfang/projects_v2/x2robot_dataset/processed_data_filtered/10009/20250705-day-make_sachets_color_modified/20250705-day-make_sachets_color_modified@MASTER_SLAVE_MODE@2025_07_05_16_40_07',
        "st_frame": 700,
        "ed_frame": 800,
    }
    agent_pos_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation",
        "follow_right_gripper",
        "velocity_decomposed",
        "height",
        "head_rotation",
        "follow_left_gripper_cur",
        "follow_right_gripper_cur"
    ]
    prediction_action_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation",
        "follow_right_gripper",
        "velocity_decomposed",
        "height",
        "head_rotation",
        "follow_left_gripper_cur",
        "follow_right_gripper_cur"
    ]
    data_config = X2RDataProcessingConfig()
    from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
    cam_mapping = _CAM_MAPPING
    data_config.update(
        cam_mapping=cam_mapping,
        class_type="x2",
        train_test_split=0.9,
        filter_angle_outliers=False,
        sample_rate=1,
        parse_tactile=False,
        obs_action_keys=agent_pos_keys,
        predict_action_keys=prediction_action_keys,
        trim_stationary=False,
        one_by_one_relative=False,
        distributed_instruction_ratio=1.0,
        cache_dir = "/x2robot_v2/xinyuanfang/projects_v2/x2robot_dataset"
    )
    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=False,
        right_padding=True,
        action_horizon=20,
        action_history_length=0,
        predict_action_keys=prediction_action_keys,
    )
    print(data_chunk_config.predict_action_keys)
    print(data_chunk_config.use_relative_action)
    check_episode_fn = _default_check_episode_fn
    get_frame_fn = _default_get_frame_fn
    # frames = get_frame_fn(episode_item, data_config, data_chunk_config)
    # print(frames[60]["action"].shape)
    # topic_item = {
    #     "path": '/x2robot/zhengwei/10055/20250526-day-fasten_the_belt',
    # }
    # result_list = check_episode_fn(topic_item, data_config, data_chunk_config)
    # print(result_list)

    topic_item = {
        "path": '/home/fangxinyuan/test',
    }
    result_list = check_episode_fn(topic_item, data_config, data_chunk_config)
    print(result_list)
    import pdb; pdb.set_trace()
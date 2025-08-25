import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device='cuda:0'

import socket
import matplotlib.pyplot as plt

from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import cv2
import torchvision
import torchvision.transforms as tv_transform
import struct
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import json
from omegaconf import OmegaConf
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import move_axis_only_real_obs_dict
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from x2robot_dataset.common.data_utils import relative_to_actions, actions_to_relative
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
from x2robot_dataset.dynamic_robot_dataset import _default_get_frame_fn
from tqdm import tqdm

def main(args):

    # Load Pretrained Policy Model
    ckpt_path = args.checkpoint_path
    print(f'ckpt_path:{ckpt_path}')
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    rgb_keys = list()
    lowdim_keys = list()
    obs_shape_meta = cfg.shape_meta.obs
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    workspace = cls(cfg, cfg.multi_run.run_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model  # type: ignore
    # if cfg.training.use_ema:
    #     policy = workspace.ema_model
    device = torch.device('cuda')
    # print(policy)
    policy.eval().to(device)

    # Load Dataset - 现在视频和数据都在同一个路径
    data_path = args.data_path
    
    # 从数据路径读取动作数据长度
    action_data_json_path = os.path.join(data_path, os.path.basename(data_path) + '.json')
    with open(action_data_json_path, 'r') as f:
        action_data = json.load(f)
    end_frame = len(action_data['data']) - 1  
    
    # 创建episode_item，使用数据路径（包含视频文件和动作数据）
    episode_item = {
        "path": data_path,
        "st_frame": 0,
        "ed_frame": end_frame,
    }
    
    # 从配置文件中获取action keys
    predict_action_keys = cfg.task.predict_action_keys
    obs_action_keys = cfg.task.obs_action_keys
    
    # 创建数据配置
    data_config = X2RDataProcessingConfig()
    cam_mapping = _CAM_MAPPING
    data_config.update(
        cam_mapping=cam_mapping,
        class_type="x2",
        train_test_split=0.9,
        filter_angle_outliers=False,
        sample_rate=1,
        parse_tactile=False,
        obs_action_keys=obs_action_keys,
        predict_action_keys=predict_action_keys,
        trim_stationary=False,
        one_by_one_relative=False,
        # 🔑 关键：设置自定义动作数据路径相关配置
        use_custom_action_data_path=True,
        global_action_data_base_path="/x2robot_v2/wjm/prj/processed_data_filtered"
    )
    
    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=False,
        right_padding=True,
        action_horizon=args.action_horizon,
        action_history_length=0,
        predict_action_keys=predict_action_keys,
    )
    
    # 使用标准的数据加载函数
    get_frame_fn = _default_get_frame_fn
    frames = get_frame_fn(episode_item, data_config, data_chunk_config)
    
    print(f"Successfully loaded {len(frames)} frames")
    print(f"Frame 0 obs keys: {list(frames[0]['obs'].keys())}")
    obs_shapes = {k: v.shape for k, v in frames[0]['obs'].items()}
    print(f"Frame 0 obs shapes: {obs_shapes}")

    # Model Inference Compare to Action Data
    ground_truths = []
    model_predicts = []
    for i in tqdm(range(0, len(frames), args.action_horizon)):
        obs_dict = frames[i]
        obs = obs_dict['obs']
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            obs_dict_normalized = dict_apply(obs, lambda x: x.unsqueeze(0).to(device))
            # 修复：包装观察数据在'obs'键中
            batch = {
                'obs': obs_dict_normalized
            }
            result = policy.predict_action(batch)
            action_pred = result['action_pred'][0].detach().to('cpu').numpy()
            # import pdb; pdb.set_trace()
        model_predicts.append(action_pred)
        ground_truths.append(frames[i]['action'])

    ground_truths = np.concatenate(ground_truths, axis=0)
    model_predicts = np.concatenate(model_predicts, axis=0)

    dim = ground_truths.shape[-1]

    ground_truths = ground_truths.reshape(-1, dim)
    model_predicts = model_predicts.reshape(-1, dim)

    # 创建动作维度标签映射 (22维动作)
    action_labels = [
        # Left arm (7 dimensions)
        'Left Arm X', 'Left Arm Y', 'Left Arm Z',           # follow_left_ee_cartesian_pos (3)
        'Left Arm Roll', 'Left Arm Pitch', 'Left Arm Yaw',  # follow_left_ee_rotation (3)
        'Left Gripper',                                      # follow_left_gripper (1)
        # Right arm (7 dimensions)  
        'Right Arm X', 'Right Arm Y', 'Right Arm Z',        # follow_right_ee_cartesian_pos (3)
        'Right Arm Roll', 'Right Arm Pitch', 'Right Arm Yaw', # follow_right_ee_rotation (3)
        'Right Gripper',                                     # follow_right_gripper (1)
        # Mobile base (3 dimensions)
        'Base Vel X', 'Base Vel Y', 'Base Angular Vel',     # base_velocity (3)
        # Height (1 dimension)
        'Height',                                            # height (1)
        # Head rotation (2 dimensions)
        'Head Pan', 'Head Tilt',                            # head_rotation (2)
        # Gripper current (2 dimensions)
        'Left Gripper Current', 'Right Gripper Current'     # gripper current (2)
    ]
    
    # 确保标签数量与动作维度匹配
    if len(action_labels) != dim:
        print(f"Warning: Expected {dim} action dimensions, but have {len(action_labels)} labels")
        # 如果标签数量不匹配，使用通用标签
        action_labels = [f'Action Dim {i+1}' for i in range(dim)]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 3*dim))

    for i in range(dim):
        plt.subplot(dim, 1, i + 1)

        # plot every 10th action
        plt.xticks(np.arange(0, len(ground_truths), step=10))

        plt.plot(ground_truths[:, i], label='Ground Truth', color='blue', linewidth=1.5)
        plt.plot(model_predicts[:, i], label='Model Output', color='orange', linewidth=1.5)
        plt.title(f'{action_labels[i]} (Dim {i+1})', fontsize=12, fontweight='bold')
        plt.xlabel('Time Step')
        plt.ylabel('Action Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('openloop_action_comparison1.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'openloop_action_compariso1.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/x2robot_v2/wjm/prj/MoMaTwin/policy/Mobile-DP/logs/2025.07.04/20.52.13_navigate_and_pick_waste_navigate_and_pick_waste/checkpoints/epoch=0082-train_loss=0.00041.ckpt")
    parser.add_argument("--action_horizon", type=int, default=40)
    parser.add_argument("--data_path", type=str, default="/x2robot_v2/wjm/prj/processed_data_filtered/10103/20250704-day-navigate_and_pick_waste-dual-turtle/20250704-day-navigate_and_pick_waste-dual-turtle@MASTER_SLAVE_MODE@2025_07_04_10_49_14", help="Path containing both video files and action data")
    args = parser.parse_args()
    main(args)

    
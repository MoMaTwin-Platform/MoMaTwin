import os
import torch
from tqdm import tqdm
import time
from accelerate import Accelerator
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dynamic_robot_dataset import DynamicRobotDataset
from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
from x2robot_dataset.common.collate_fn import collate_wrapper

def tensor_to_strings(tensor):
    """
    将 shape 为 [B, L] 的 int32 tensor 转换为字符串列表。
    每行代表一个样本，去除结尾的 0，并尝试解码为 UTF-8 字符串。
    """
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
        return []

    batch_size, max_len = tensor.shape
    strings = []

    for i in range(batch_size):
        # 取出非零的部分
        indices = tensor[i].tolist()
        zero_pos = indices.index(0) if 0 in indices else max_len
        chars = [chr(c) for c in indices[:zero_pos]]
        s = ''.join(chars)
        strings.append(s)

    return strings

def print_tensor_shapes(data, indent=0):
    """
    递归打印嵌套字典中所有 tensor 的 shape。
    支持 dict、list、tuple 等结构，并特别处理 dataset_name key。
    """
    import torch

    prefix = "  " * indent

    if isinstance(data, dict):
        for k, v in data.items():
            if k == 'dataset_name' and isinstance(v, torch.Tensor):
                # 特别处理 dataset_name，打印每一行字符串
                print(f"{prefix}Key: '{k}'")
                print(f"{prefix}  Tensor shape: {v.shape}")
                strings = tensor_to_strings(v)
                for idx, s in enumerate(strings):
                    print(f"{prefix}    Sample {idx}: {s}")
            else:
                print(f"{prefix}Key: '{k}'")
                print_tensor_shapes(v, indent + 1)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            print(f"{prefix}Index: {i}")
            print_tensor_shapes(item, indent + 1)
    elif isinstance(data, torch.Tensor):
        print(f"{prefix}Tensor shape: {data.shape}")
    else:
        print(f"{prefix}Value: {type(data)}")

def main():
    print(os.environ['MASTER_ADDR'])
    print(os.environ['MASTER_PORT'])
    accelerator = Accelerator()
    dataset_config_path = "/x2robot/liuxingchen/code/diffusion_policy/diffusion_policy/config/task/entangle_line_threefork.yaml"
    # Example usage
    
    _ACTION_KEY_FULL_MAPPING_XY = {
        "follow_right_arm_joint_pos": "follow_right_joint_pos",
        "follow_right_arm_joint_dev": "follow_right_joint_dev",
        "follow_right_arm_joint_cur": "follow_right_joint_cur",
        "follow_right_ee_cartesian_pos": "follow_right_position",
        "follow_right_ee_rotation": "follow_right_rotation",
        "follow_right_gripper": "follow_right_gripper",
        "master_right_ee_cartesian_pos": "master_right_position",
        "master_right_ee_rotation": "master_right_rotation",
        "master_right_gripper": "master_right_gripper",
        "follow_left_arm_joint_pos": "follow_left_joint_pos",
        "follow_left_arm_joint_dev": "follow_left_joint_dev",
        "follow_left_arm_joint_cur": "follow_left_joint_cur",
        "follow_left_ee_cartesian_pos": "follow_left_position",
        "follow_left_ee_rotation": "follow_left_rotation",
        "follow_left_gripper": "follow_left_gripper",
        "master_left_ee_cartesian_pos": "master_left_position",
        "master_left_ee_rotation": "master_left_rotation",
        "master_left_gripper": "master_left_gripper",
    }

    prediction_action_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation",
        "follow_left_gripper",
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation",
        "follow_right_gripper",
        'follow_left_arm_joint_cur',
        'follow_right_arm_joint_cur',
    ]
    data_config = X2RDataProcessingConfig()
    data_config.update(
        cam_mapping=_CAM_MAPPING,
        class_type="x2",
        train_test_split=0.95,
        filter_angle_outliers=False,
        sample_rate=1,
        parse_tactile=False,
        action_keys=list(_ACTION_KEY_FULL_MAPPING_XY.keys()),
        predict_action_keys=prediction_action_keys,
        use_gripper_cur=True,
        trim_stationary=True,
    )

    data_chunk_config = X2RDataChunkConfig().update(
        left_padding=False,
        right_padding=True,
        action_horizon=21,
        action_history_length=0,
    )
    dataset = DynamicRobotDataset(
        dataset_config_path=dataset_config_path,
        data_config=data_config,
        data_chunk_config=data_chunk_config,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        batch_size=32,
    )

    epochs = 1000
    # 训练循环
    for epoch in range(epochs):
        train_loader = dataset.get_train_dataloader()
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{epochs} 开始， 共 {len(train_loader)} 个批次")
        start_time = time.time()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training", total=len(train_loader), disable=not accelerator.is_main_process)):
            if accelerator.is_main_process and (batch_idx + 1) % 100 == 0:
                print(f"\n批次抽检 {batch_idx}:")
                print_tensor_shapes(batch)
            if accelerator.is_main_process:
                elapsed = time.time() - start_time
                speed = (batch_idx + 1) / elapsed
                if (batch_idx + 1) % 10 == 0:  # 每10个batch更新一次速度
                    print(f" ➤ 已处理 {batch_idx + 1} 个 batch，平均速度: {speed:.2f} batch/s")
            accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()  
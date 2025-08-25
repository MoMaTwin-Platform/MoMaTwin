"""
Instruction处理模块
负责处理各种类型的instruction数据，包括从文件加载、格式转换、subtask处理等
"""

import os
import json
import random
from typing import Dict, Any, Optional, Tuple, List
from x2robot_dataset.lazy_dataset import X2RDataProcessingConfig


class InstructionProcessor:
    """
    instruction处理器，负责统一处理各种instruction相关的逻辑
    """
    
    def __init__(self, data_config: X2RDataProcessingConfig):
        """
        初始化instruction处理器
        
        参数:
        data_config (X2RDataProcessingConfig): 数据配置对象
        """
        self.data_config = data_config
        # 添加缓存避免重复加载同一路径的instruction
        self._instruction_cache = {}
        
    def load_instruction_data(self, sub_dir_path: str) -> Dict[str, Any]:
        """
        从指定路径加载instruction数据
        
        参数:
        sub_dir_path (str): 子目录路径
        
        返回:
        dict: 包含instruction相关信息的字典
        """
        instruction_info = {}
        
        # 解析路径信息
        robot_id, task_id, sample_name = self._parse_path_info(sub_dir_path)
        
        # 处理instruction_path
        if self.data_config.instruction_path is not None:
            instruction_path = self.data_config.instruction_path.format(
                robot_id=robot_id, 
                topic_id=task_id, 
                uid=sample_name
            )
            
            if os.path.exists(instruction_path):
                try:
                    with open(instruction_path, 'r') as f:
                        loaded_instruction_info = json.load(f)
                        instruction_info.update(loaded_instruction_info)
                    print(f"Loaded instruction from {instruction_path}")
                except Exception as e:
                    print(f"Error loading instruction from {instruction_path}: {e}")
        
        # 如果没有从文件加载到instruction，使用default_instruction
        if "instruction" not in instruction_info and self.data_config.default_instruction:
            instruction_info["instruction"] = self.data_config.default_instruction
        
        return instruction_info
    
    def get_frame_instruction(self, instruction_info: Dict[str, Any], frame_idx: int) -> str:
        """
        根据配置的instruction_key和概率获取特定帧的instruction字符串
        
        参数:
        instruction_info (dict): 从load_instruction_data获得的instruction信息
        frame_idx (int): 当前帧索引
        
        返回:
        str: instruction字符串
        """
        # 获取instruction_key配置
        instruction_key_configs = getattr(self.data_config, 'instruction_key', None)
        
        if instruction_key_configs is None or not instruction_key_configs:
            # 如果没有配置instruction_key，使用默认的instruction
            return instruction_info.get("instruction", self.data_config.default_instruction or "")
        
        # 遍历所有配置的instruction_key
        for config in instruction_key_configs:
            for instruction_key, probability in config.items():
                # 根据概率决定是否使用这个instruction_key
                if random.random() < probability:
                    instruction_text = self._get_instruction_by_key(
                        instruction_info, frame_idx, instruction_key
                    )
                    if instruction_text:  # 如果成功获取到instruction
                        return instruction_text
        
        # 如果所有概率都没中，或者没有获取到有效的instruction，使用default_instruction
        return instruction_info.get("instruction", self.data_config.default_instruction or "")
    
    def _get_instruction_by_key(self, instruction_info: Dict[str, Any], frame_idx: int, 
                              instruction_key: str) -> Optional[str]:
        """
        根据指定的key获取instruction
        
        参数:
        instruction_info (dict): instruction信息
        frame_idx (int): 当前帧索引
        instruction_key (str): instruction key
        
        返回:
        str或None: instruction字符串，如果没有找到则返回None
        """
        if instruction_key == "subtask_generation":
            subtask_generation = instruction_info.get("subtask_generation", None)
            if subtask_generation is not None:
                current_subtask = self._find_current_subtask(subtask_generation, frame_idx)
                return current_subtask
        
        # 对于其他key，直接获取
        return instruction_info.get(instruction_key, None)
    
    def _parse_path_info(self, sub_dir_path: str) -> Tuple[str, str, str]:
        """
        从路径中解析robot_id, task_id, sample_name
        
        参数:
        sub_dir_path (str): 子目录路径
        
        返回:
        tuple: (robot_id, task_id, sample_name)
        """
        path_parts = [s for s in sub_dir_path.split("/")[-3:] if s != ""]
        if len(path_parts) >= 3:
            return path_parts[0], path_parts[1], path_parts[2]
        else:
            # 处理路径解析失败的情况
            return "unknown", "unknown", "unknown"
    
    def _find_current_subtask(self, subtask_generation: Dict[str, str], frame_idx: int) -> Optional[str]:
        """
        找到当前帧对应的subtask
        
        参数:
        subtask_generation (dict): subtask_generation字典
        frame_idx (int): 当前帧索引
        
        返回:
        str或None: 当前帧对应的subtask描述，如果没有则返回None
        """
        for time_range, subtask_desc in subtask_generation.items():
            try:
                start_frame, end_frame = map(int, time_range.split())
                if start_frame <= frame_idx < end_frame:
                    return subtask_desc
            except ValueError:
                # 跳过无法解析的时间范围
                continue
        return None

    def get_frame_instruction_from_path(self, sub_dir_path: str, frame_idx: int) -> str:
        """
        根据路径和帧索引获取instruction
        """
        # 检查缓存
        if sub_dir_path not in self._instruction_cache:
            self._instruction_cache[sub_dir_path] = self.load_instruction_data(sub_dir_path)
        
        instruction_info = self._instruction_cache[sub_dir_path]
        return self.get_frame_instruction(instruction_info, frame_idx)

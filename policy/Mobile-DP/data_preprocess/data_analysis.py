#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析工具 - data_analysis.py

功能：
1. 遍历处理好的数据集目录
2. 全局统计 car_pose 和 velocity_decomposed 的最大值/最小值/平均值/标准差
3. 绘制 velocity_decomposed 各个分量（vx, vy, vyaw）的速度分布直方图
4. 生成详细的统计报告

作者: @yinyuehao
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import argparse
import glob
from collections import defaultdict
import pandas as pd

# 设置非交互模式
matplotlib.use('Agg')

class DatasetAnalyzer:
    """
    数据集分析器 - 用于分析处理后的轨迹数据
    """
    
    def __init__(self):
        """初始化分析器"""
        self.car_pose_data = {
            'x': [],
            'y': [],
            'angle': []
        }
        self.velocity_data = {
            'vx': [],
            'vy': [],
            'vyaw': []
        }
        self.trajectory_stats = []
        self.total_trajectories = 0
        self.total_frames = 0
        
        # 🎯 新增：异常高速轨迹跟踪
        self.abnormal_speed_trajectories = []  # 包含异常高速的轨迹信息
        
    def find_processed_json_files(self, processed_data_dir):
        """
        递归查找所有处理过的JSON文件
        
        Args:
            processed_data_dir: 处理数据的根目录
            
        Returns:
            JSON文件路径列表
        """
        json_files = []
        
        # 遍历目录结构：processed_data/{robot_id}/{task_name}/{trajectory_name}/{trajectory_name}.json
        if not os.path.exists(processed_data_dir):
            print(f"❌ 目录不存在: {processed_data_dir}")
            return json_files
        
        # 查找模式：*/*/*.json (跳过trajectory.png等文件)
        search_pattern = os.path.join(processed_data_dir, "*", "*", "*", "*.json")
        potential_files = glob.glob(search_pattern)
        
        for file_path in potential_files:
            # 确保是轨迹JSON文件（不是其他配置文件）
            if file_path.endswith('.json'):
                # 检查文件名是否与父目录名称匹配（轨迹文件特征）
                file_name = os.path.basename(file_path)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                
                if file_name.replace('.json', '') == parent_dir:
                    json_files.append(file_path)
        
        print(f"🔍 找到 {len(json_files)} 个处理过的JSON文件")
        return sorted(json_files)
    
    def load_and_analyze_json(self, json_path):
        """
        加载并分析单个JSON文件
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            分析结果字典或None（如果失败）
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            frames = data.get('data', [])
            if not frames:
                print(f"⚠️  空数据文件: {json_path}")
                return None
            
            # 提取car_pose和velocity_decomposed数据
            trajectory_car_pose = {
                'x': [],
                'y': [],
                'angle': []
            }
            trajectory_velocity = {
                'vx': [],
                'vy': [],
                'vyaw': []
            }
            
            for frame in frames:
                # 提取car_pose数据
                car_pose = frame.get('car_pose', [0, 0, 0])
                if len(car_pose) >= 3:
                    trajectory_car_pose['x'].append(car_pose[0])
                    trajectory_car_pose['y'].append(car_pose[1])
                    trajectory_car_pose['angle'].append(car_pose[2])
                
                # 提取velocity_decomposed数据
                velocity_decomposed = frame.get('velocity_decomposed', [0, 0, 0])
                if len(velocity_decomposed) >= 3:
                    trajectory_velocity['vx'].append(velocity_decomposed[0])
                    trajectory_velocity['vy'].append(velocity_decomposed[1])
                    trajectory_velocity['vyaw'].append(velocity_decomposed[2])
            
            # 转换为numpy数组以便计算统计信息
            for key in trajectory_car_pose:
                trajectory_car_pose[key] = np.array(trajectory_car_pose[key])
            for key in trajectory_velocity:
                trajectory_velocity[key] = np.array(trajectory_velocity[key])
            
            # 🎯 新增：检测异常高速
            abnormal_speed_info = self.detect_abnormal_speed(trajectory_velocity, json_path)
            
            # 计算轨迹统计信息
            trajectory_stat = {
                'file_path': json_path,
                'trajectory_name': os.path.basename(json_path).replace('.json', ''),
                'num_frames': len(frames),
                'car_pose_stats': {},
                'velocity_stats': {},
                'abnormal_speed_info': abnormal_speed_info  # 🎯 新增异常高速信息
            }
            
            # Car pose统计
            for key, values in trajectory_car_pose.items():
                if len(values) > 0:
                    trajectory_stat['car_pose_stats'][key] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
            
            # Velocity统计
            for key, values in trajectory_velocity.items():
                if len(values) > 0:
                    trajectory_stat['velocity_stats'][key] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'range': float(np.max(values) - np.min(values))
                    }
            
            return {
                'trajectory_stat': trajectory_stat,
                'car_pose_data': trajectory_car_pose,
                'velocity_data': trajectory_velocity
            }
            
        except Exception as e:
            print(f"❌ 加载JSON文件失败: {json_path}, 错误: {e}")
            return None
    
    def detect_abnormal_speed(self, trajectory_velocity, json_path):
        """
        检测轨迹中的异常高速情况
        
        Args:
            trajectory_velocity: 轨迹速度数据
            json_path: JSON文件路径
            
        Returns:
            异常高速信息字典
        """
        # 异常高速阈值
        LINEAR_SPEED_THRESHOLD = 0.5  # m/s
        ANGULAR_SPEED_THRESHOLD = 0.6  # rad/s
        
        abnormal_info = {
            'has_abnormal_speed': False,
            'abnormal_types': [],
            'max_speeds': {},
            'abnormal_frame_counts': {},
            'abnormal_frame_percentages': {}
        }
        
        vx_data = trajectory_velocity.get('vx', np.array([]))
        vy_data = trajectory_velocity.get('vy', np.array([]))
        vyaw_data = trajectory_velocity.get('vyaw', np.array([]))
        
        total_frames = len(vx_data)
        
        # 检测vx异常高速
        if len(vx_data) > 0:
            vx_abnormal = np.abs(vx_data) > LINEAR_SPEED_THRESHOLD
            abnormal_info['max_speeds']['vx'] = float(np.max(np.abs(vx_data)))
            abnormal_info['abnormal_frame_counts']['vx'] = int(np.sum(vx_abnormal))
            abnormal_info['abnormal_frame_percentages']['vx'] = float(np.sum(vx_abnormal) / total_frames * 100)
            
            if np.any(vx_abnormal):
                abnormal_info['has_abnormal_speed'] = True
                abnormal_info['abnormal_types'].append('vx')
        
        # 检测vy异常高速
        if len(vy_data) > 0:
            vy_abnormal = np.abs(vy_data) > LINEAR_SPEED_THRESHOLD
            abnormal_info['max_speeds']['vy'] = float(np.max(np.abs(vy_data)))
            abnormal_info['abnormal_frame_counts']['vy'] = int(np.sum(vy_abnormal))
            abnormal_info['abnormal_frame_percentages']['vy'] = float(np.sum(vy_abnormal) / total_frames * 100)
            
            if np.any(vy_abnormal):
                abnormal_info['has_abnormal_speed'] = True
                abnormal_info['abnormal_types'].append('vy')
        
        # 检测vyaw异常高速
        if len(vyaw_data) > 0:
            vyaw_abnormal = np.abs(vyaw_data) > ANGULAR_SPEED_THRESHOLD
            abnormal_info['max_speeds']['vyaw'] = float(np.max(np.abs(vyaw_data)))
            abnormal_info['abnormal_frame_counts']['vyaw'] = int(np.sum(vyaw_abnormal))
            abnormal_info['abnormal_frame_percentages']['vyaw'] = float(np.sum(vyaw_abnormal) / total_frames * 100)
            
            if np.any(vyaw_abnormal):
                abnormal_info['has_abnormal_speed'] = True
                abnormal_info['abnormal_types'].append('vyaw')
        
        # 检测速度幅值异常高速
        if len(vx_data) > 0 and len(vy_data) > 0:
            speed_magnitude = np.sqrt(vx_data**2 + vy_data**2)
            speed_abnormal = speed_magnitude > LINEAR_SPEED_THRESHOLD
            abnormal_info['max_speeds']['speed_magnitude'] = float(np.max(speed_magnitude))
            abnormal_info['abnormal_frame_counts']['speed_magnitude'] = int(np.sum(speed_abnormal))
            abnormal_info['abnormal_frame_percentages']['speed_magnitude'] = float(np.sum(speed_abnormal) / total_frames * 100)
            
            if np.any(speed_abnormal):
                abnormal_info['has_abnormal_speed'] = True
                if 'speed_magnitude' not in abnormal_info['abnormal_types']:
                    abnormal_info['abnormal_types'].append('speed_magnitude')
        
        return abnormal_info
    
    def analyze_all_data(self, processed_data_dir):
        """
        分析所有处理过的数据
        
        Args:
            processed_data_dir: 处理数据的根目录
        """
        print(f"🔍 开始分析数据目录: {processed_data_dir}")
        
        # 查找所有JSON文件
        json_files = self.find_processed_json_files(processed_data_dir)
        
        if not json_files:
            print("❌ 未找到任何处理过的JSON文件")
            return
        
        # 重置数据
        self.car_pose_data = {'x': [], 'y': [], 'angle': []}
        self.velocity_data = {'vx': [], 'vy': [], 'vyaw': []}
        self.trajectory_stats = []
        self.abnormal_speed_trajectories = []  # 🎯 重置异常高速轨迹列表
        self.total_trajectories = 0
        self.total_frames = 0
        
        # 逐个分析JSON文件
        print("📊 正在分析轨迹数据...")
        for json_path in tqdm(json_files, desc="分析进度"):
            result = self.load_and_analyze_json(json_path)
            
            if result is not None:
                # 累积全局数据
                for key in self.car_pose_data:
                    self.car_pose_data[key].extend(result['car_pose_data'][key])
                
                for key in self.velocity_data:
                    self.velocity_data[key].extend(result['velocity_data'][key])
                
                # 保存轨迹统计
                trajectory_stat = result['trajectory_stat']
                self.trajectory_stats.append(trajectory_stat)
                
                # 🎯 新增：记录异常高速轨迹
                if trajectory_stat['abnormal_speed_info']['has_abnormal_speed']:
                    self.abnormal_speed_trajectories.append({
                        'file_path': trajectory_stat['file_path'],
                        'trajectory_name': trajectory_stat['trajectory_name'],
                        'abnormal_types': trajectory_stat['abnormal_speed_info']['abnormal_types'],
                        'max_speeds': trajectory_stat['abnormal_speed_info']['max_speeds'],
                        'abnormal_frame_counts': trajectory_stat['abnormal_speed_info']['abnormal_frame_counts'],
                        'abnormal_frame_percentages': trajectory_stat['abnormal_speed_info']['abnormal_frame_percentages']
                    })
                
                self.total_trajectories += 1
                self.total_frames += trajectory_stat['num_frames']
        
        # 转换为numpy数组
        for key in self.car_pose_data:
            self.car_pose_data[key] = np.array(self.car_pose_data[key])
        for key in self.velocity_data:
            self.velocity_data[key] = np.array(self.velocity_data[key])
        
        print(f"✅ 分析完成!")
        print(f"📈 总计: {self.total_trajectories} 条轨迹, {self.total_frames} 帧数据")
        print(f"⚠️  异常高速轨迹: {len(self.abnormal_speed_trajectories)} 条 ({len(self.abnormal_speed_trajectories)/max(self.total_trajectories, 1)*100:.1f}%)")
    
    def calculate_global_statistics(self):
        """
        计算全局统计信息
        
        Returns:
            全局统计字典
        """
        global_stats = {
            'summary': {
                'total_trajectories': self.total_trajectories,
                'total_frames': self.total_frames,
                'avg_frames_per_trajectory': self.total_frames / max(self.total_trajectories, 1)
            },
            'car_pose': {},
            'velocity_decomposed': {}
        }
        
        # Car pose全局统计
        for key, values in self.car_pose_data.items():
            if len(values) > 0:
                global_stats['car_pose'][key] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'percentile_25': float(np.percentile(values, 25)),
                    'percentile_75': float(np.percentile(values, 75)),
                    'range': float(np.max(values) - np.min(values)),
                    'total_samples': len(values)
                }
        
        # Velocity全局统计
        for key, values in self.velocity_data.items():
            if len(values) > 0:
                global_stats['velocity_decomposed'][key] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'percentile_25': float(np.percentile(values, 25)),
                    'percentile_75': float(np.percentile(values, 75)),
                    'range': float(np.max(values) - np.min(values)),
                    'total_samples': len(values)
                }
        
        return global_stats
    
    def calculate_distribution_percentages(self):
        """
        计算数据分布百分比
        
        Returns:
            分布百分比字典
        """
        distribution_stats = {
            'velocity_distribution': {},
            'car_pose_distribution': {},
            'abnormal_speed_analysis': {}  # 🎯 新增异常高速分析
        }
        
        # 🎯 新增：异常高速分析
        LINEAR_SPEED_THRESHOLD = 0.5  # m/s
        ANGULAR_SPEED_THRESHOLD = 0.6  # rad/s
        
        abnormal_speed_stats = {
            'linear_speed_threshold': LINEAR_SPEED_THRESHOLD,
            'angular_speed_threshold': ANGULAR_SPEED_THRESHOLD,
            'total_trajectories': self.total_trajectories,
            'abnormal_trajectories_count': len(self.abnormal_speed_trajectories),
            'abnormal_trajectories_percentage': len(self.abnormal_speed_trajectories) / max(self.total_trajectories, 1) * 100,
            'abnormal_trajectory_details': self.abnormal_speed_trajectories
        }
        
        # 全局异常高速帧统计
        global_abnormal_frames = {
            'vx': 0, 'vy': 0, 'vyaw': 0, 'speed_magnitude': 0
        }
        
        # 计算全局异常高速帧数
        if 'vx' in self.velocity_data and len(self.velocity_data['vx']) > 0:
            global_abnormal_frames['vx'] = int(np.sum(np.abs(self.velocity_data['vx']) > LINEAR_SPEED_THRESHOLD))
        if 'vy' in self.velocity_data and len(self.velocity_data['vy']) > 0:
            global_abnormal_frames['vy'] = int(np.sum(np.abs(self.velocity_data['vy']) > LINEAR_SPEED_THRESHOLD))
        if 'vyaw' in self.velocity_data and len(self.velocity_data['vyaw']) > 0:
            global_abnormal_frames['vyaw'] = int(np.sum(np.abs(self.velocity_data['vyaw']) > ANGULAR_SPEED_THRESHOLD))
        if 'vx' in self.velocity_data and 'vy' in self.velocity_data:
            vx_data = self.velocity_data['vx']
            vy_data = self.velocity_data['vy']
            if len(vx_data) > 0 and len(vy_data) > 0:
                speed_magnitude = np.sqrt(vx_data**2 + vy_data**2)
                global_abnormal_frames['speed_magnitude'] = int(np.sum(speed_magnitude > LINEAR_SPEED_THRESHOLD))
        
        abnormal_speed_stats['global_abnormal_frame_counts'] = global_abnormal_frames
        abnormal_speed_stats['global_abnormal_frame_percentages'] = {
            key: count / max(self.total_frames, 1) * 100 
            for key, count in global_abnormal_frames.items()
        }
        
        distribution_stats['abnormal_speed_analysis'] = abnormal_speed_stats
        
        # 原有的速度分布分析
        for key in ['vx', 'vy', 'vyaw']:
            if key in self.velocity_data:
                data = self.velocity_data[key]
                total_samples = len(data)
                
                if total_samples > 0:
                    # 计算不同范围的占比
                    near_zero = np.abs(data) < 0.01  # 接近静止
                    low_speed = (np.abs(data) >= 0.01) & (np.abs(data) < 0.1)  # 低速
                    medium_speed = (np.abs(data) >= 0.1) & (np.abs(data) < 0.3)  # 中速
                    high_speed = (np.abs(data) >= 0.3) & (np.abs(data) < (LINEAR_SPEED_THRESHOLD if key in ['vx', 'vy'] else ANGULAR_SPEED_THRESHOLD))  # 高速
                    abnormal_high_speed = np.abs(data) >= (LINEAR_SPEED_THRESHOLD if key in ['vx', 'vy'] else ANGULAR_SPEED_THRESHOLD)  # 🎯 新增异常高速
                    
                    # 正负方向分析
                    positive_vals = data > 0.01
                    negative_vals = data < -0.01
                    
                    # 异常值分析（超过3个标准差）
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    outliers = np.abs(data - mean_val) > 3 * std_val
                    
                    distribution_stats['velocity_distribution'][key] = {
                        'near_zero_pct': float(np.sum(near_zero) / total_samples * 100),
                        'low_speed_pct': float(np.sum(low_speed) / total_samples * 100),
                        'medium_speed_pct': float(np.sum(medium_speed) / total_samples * 100),
                        'high_speed_pct': float(np.sum(high_speed) / total_samples * 100),
                        'abnormal_high_speed_pct': float(np.sum(abnormal_high_speed) / total_samples * 100),  # 🎯 新增
                        'positive_pct': float(np.sum(positive_vals) / total_samples * 100),
                        'negative_pct': float(np.sum(negative_vals) / total_samples * 100),
                        'outliers_pct': float(np.sum(outliers) / total_samples * 100),
                        'total_samples': total_samples
                    }
        
        # Car pose分布分析（保持原有逻辑）
        for key in ['x', 'y']:
            if key in self.car_pose_data:
                data = self.car_pose_data[key]
                total_samples = len(data)
                
                if total_samples > 0:
                    # 计算不同范围的占比
                    near_origin = np.abs(data) < 0.1  # 接近原点
                    small_range = (np.abs(data) >= 0.1) & (np.abs(data) < 1.0)  # 小范围
                    medium_range = (np.abs(data) >= 1.0) & (np.abs(data) < 3.0)  # 中等范围
                    large_range = np.abs(data) >= 3.0  # 大范围
                    
                    # 正负方向分析
                    positive_vals = data > 0.1
                    negative_vals = data < -0.1
                    
                    # 异常值分析
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    outliers = np.abs(data - mean_val) > 3 * std_val
                    
                    distribution_stats['car_pose_distribution'][key] = {
                        'near_origin_pct': float(np.sum(near_origin) / total_samples * 100),
                        'small_range_pct': float(np.sum(small_range) / total_samples * 100),
                        'medium_range_pct': float(np.sum(medium_range) / total_samples * 100),
                        'large_range_pct': float(np.sum(large_range) / total_samples * 100),
                        'positive_pct': float(np.sum(positive_vals) / total_samples * 100),
                        'negative_pct': float(np.sum(negative_vals) / total_samples * 100),
                        'outliers_pct': float(np.sum(outliers) / total_samples * 100),
                        'total_samples': total_samples
                    }
        
        # 角度分布分析（保持原有逻辑）
        if 'angle' in self.car_pose_data:
            angle_data = self.car_pose_data['angle']
            total_samples = len(angle_data)
            
            if total_samples > 0:
                # 象限分析（考虑角度的周期性）
                angle_normalized = (angle_data % (2 * np.pi))  # 标准化到[0, 2π]
                
                q1 = (angle_normalized >= 0) & (angle_normalized < np.pi/2)  # 第一象限
                q2 = (angle_normalized >= np.pi/2) & (angle_normalized < np.pi)  # 第二象限
                q3 = (angle_normalized >= np.pi) & (angle_normalized < 3*np.pi/2)  # 第三象限
                q4 = (angle_normalized >= 3*np.pi/2) & (angle_normalized < 2*np.pi)  # 第四象限
                
                # 角度变化范围分析
                angle_range = np.max(angle_data) - np.min(angle_data)
                
                # 异常值分析
                mean_val = np.mean(angle_data)
                std_val = np.std(angle_data)
                outliers = np.abs(angle_data - mean_val) > 3 * std_val
                
                distribution_stats['car_pose_distribution']['angle'] = {
                    'quadrant_1_pct': float(np.sum(q1) / total_samples * 100),
                    'quadrant_2_pct': float(np.sum(q2) / total_samples * 100),
                    'quadrant_3_pct': float(np.sum(q3) / total_samples * 100),
                    'quadrant_4_pct': float(np.sum(q4) / total_samples * 100),
                    'angle_range_rad': float(angle_range),
                    'angle_range_deg': float(angle_range * 180 / np.pi),
                    'outliers_pct': float(np.sum(outliers) / total_samples * 100),
                    'total_samples': total_samples
                }
        
        # 速度幅值分布分析
        if 'vx' in self.velocity_data and 'vy' in self.velocity_data:
            vx_data = self.velocity_data['vx']
            vy_data = self.velocity_data['vy']
            if len(vx_data) > 0 and len(vy_data) > 0:
                speed_magnitude = np.sqrt(vx_data**2 + vy_data**2)
                total_samples = len(speed_magnitude)
                
                # 速度幅值范围分析
                stationary = speed_magnitude < 0.01  # 静止
                slow = (speed_magnitude >= 0.01) & (speed_magnitude < 0.1)  # 缓慢
                normal = (speed_magnitude >= 0.1) & (speed_magnitude < 0.4)  # 正常
                fast = (speed_magnitude >= 0.4) & (speed_magnitude < LINEAR_SPEED_THRESHOLD)  # 快速
                abnormal_fast = speed_magnitude >= LINEAR_SPEED_THRESHOLD  # 🎯 新增异常快速
                
                distribution_stats['velocity_distribution']['speed_magnitude'] = {
                    'stationary_pct': float(np.sum(stationary) / total_samples * 100),
                    'slow_pct': float(np.sum(slow) / total_samples * 100),
                    'normal_pct': float(np.sum(normal) / total_samples * 100),
                    'fast_pct': float(np.sum(fast) / total_samples * 100),
                    'abnormal_fast_pct': float(np.sum(abnormal_fast) / total_samples * 100),  # 🎯 新增
                    'max_speed': float(np.max(speed_magnitude)),
                    'avg_speed': float(np.mean(speed_magnitude)),
                    'total_samples': total_samples
                }
        
        return distribution_stats
    
    def create_velocity_histograms(self, output_dir, bins=50):
        """
        创建速度分布直方图
        
        Args:
            output_dir: 输出目录
            bins: 直方图bin数量
        """
        print("📊 正在生成速度分布直方图...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Velocity Distribution Analysis', fontsize=16, fontweight='bold')
        
        # vx分布直方图
        ax = axes[0, 0]
        vx_data = self.velocity_data['vx']
        if len(vx_data) > 0:
            n, bins_vx, patches = ax.hist(vx_data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(vx_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vx_data):.3f}')
            ax.axvline(np.median(vx_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(vx_data):.3f}')
        ax.set_title('Forward Velocity (vx) Distribution', fontweight='bold')
        ax.set_xlabel('vx (m/s)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # vy分布直方图
        ax = axes[0, 1]
        vy_data = self.velocity_data['vy']
        if len(vy_data) > 0:
            n, bins_vy, patches = ax.hist(vy_data, bins=bins, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.mean(vy_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vy_data):.3f}')
            ax.axvline(np.median(vy_data), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(vy_data):.3f}')
        ax.set_title('Lateral Velocity (vy) Distribution', fontweight='bold')
        ax.set_xlabel('vy (m/s)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # vyaw分布直方图
        ax = axes[1, 0]
        vyaw_data = self.velocity_data['vyaw']
        if len(vyaw_data) > 0:
            n, bins_vyaw, patches = ax.hist(vyaw_data, bins=bins, alpha=0.7, color='red', edgecolor='black')
            ax.axvline(np.mean(vyaw_data), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(vyaw_data):.3f}')
            ax.axvline(np.median(vyaw_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(vyaw_data):.3f}')
        ax.set_title('Angular Velocity (vyaw) Distribution', fontweight='bold')
        ax.set_xlabel('vyaw (rad/s)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 速度幅值分布直方图
        ax = axes[1, 1]
        if len(vx_data) > 0 and len(vy_data) > 0:
            speed_magnitude = np.sqrt(vx_data**2 + vy_data**2)
            n, bins_speed, patches = ax.hist(speed_magnitude, bins=bins, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(np.mean(speed_magnitude), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speed_magnitude):.3f}')
            ax.axvline(np.median(speed_magnitude), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(speed_magnitude):.3f}')
        ax.set_title('Speed Magnitude Distribution', fontweight='bold')
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存图像
        histogram_path = os.path.join(output_dir, "velocity_distribution_histograms.png")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 速度分布直方图已保存: {histogram_path}")
    
    def create_car_pose_analysis(self, output_dir):
        """
        创建car_pose分析图表
        
        Args:
            output_dir: 输出目录
        """
        print("📊 正在生成car_pose分析图表...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Car Pose Distribution Analysis', fontsize=16, fontweight='bold')
        
        # X位置分布
        ax = axes[0, 0]
        x_data = self.car_pose_data['x']
        if len(x_data) > 0:
            ax.hist(x_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(x_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(x_data):.3f}')
            ax.axvline(np.median(x_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(x_data):.3f}')
        ax.set_title('X Position Distribution', fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Y位置分布
        ax = axes[0, 1]
        y_data = self.car_pose_data['y']
        if len(y_data) > 0:
            ax.hist(y_data, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(np.mean(y_data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y_data):.3f}')
            ax.axvline(np.median(y_data), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(y_data):.3f}')
        ax.set_title('Y Position Distribution', fontweight='bold')
        ax.set_xlabel('Y Position (m)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 角度分布
        ax = axes[1, 0]
        angle_data = self.car_pose_data['angle']
        if len(angle_data) > 0:
            ax.hist(angle_data, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax.axvline(np.mean(angle_data), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(angle_data):.3f}')
            ax.axvline(np.median(angle_data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(angle_data):.3f}')
        ax.set_title('Angle Distribution', fontweight='bold')
        ax.set_xlabel('Angle (rad)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 位置幅值分布
        ax = axes[1, 1]
        if len(x_data) > 0 and len(y_data) > 0:
            position_magnitude = np.sqrt(x_data**2 + y_data**2)
            ax.hist(position_magnitude, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax.axvline(np.mean(position_magnitude), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(position_magnitude):.3f}')
            ax.axvline(np.median(position_magnitude), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(position_magnitude):.3f}')
        ax.set_title('Position Magnitude Distribution', fontweight='bold')
        ax.set_xlabel('Distance from Origin (m)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # 保存图像
        car_pose_path = os.path.join(output_dir, "car_pose_distribution_analysis.png")
        plt.savefig(car_pose_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Car pose分析图表已保存: {car_pose_path}")
    
    def save_statistics_report(self, output_dir):
        """
        保存详细的统计报告
        
        Args:
            output_dir: 输出目录
        """
        print("📄 正在生成统计报告...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算全局统计
        global_stats = self.calculate_global_statistics()
        
        # 计算分布百分比
        distribution_stats = self.calculate_distribution_percentages()
        
        # 生成详细报告
        report_path = os.path.join(output_dir, "statistics_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'global_statistics': global_stats,
                'distribution_percentages': distribution_stats,
                'trajectory_statistics': self.trajectory_stats
            }, f, indent=2, ensure_ascii=False)
        
        # 生成人类可读的报告
        readable_report_path = os.path.join(output_dir, "statistics_report.txt")
        with open(readable_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("数据集统计分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 总体信息
            f.write("📊 总体信息:\n")
            f.write(f"  总轨迹数: {global_stats['summary']['total_trajectories']}\n")
            f.write(f"  总帧数: {global_stats['summary']['total_frames']}\n")
            f.write(f"  平均每轨迹帧数: {global_stats['summary']['avg_frames_per_trajectory']:.1f}\n\n")
            
            # 🎯 新增：异常高速分析
            f.write("📈 异常高速分析:\n")
            f.write(f"  异常高速阈值: {distribution_stats['abnormal_speed_analysis']['linear_speed_threshold']:.2f} m/s 和 {distribution_stats['abnormal_speed_analysis']['angular_speed_threshold']:.2f} rad/s\n")
            f.write(f"  异常轨迹数: {distribution_stats['abnormal_speed_analysis']['abnormal_trajectories_count']}\n")
            f.write(f"  异常轨迹百分比: {distribution_stats['abnormal_speed_analysis']['abnormal_trajectories_percentage']:.2f}%\n")
            
            # 🎯 新增：异常高速轨迹详细信息
            if len(self.abnormal_speed_trajectories) > 0:
                f.write(f"\n  异常高速轨迹详细信息:\n")
                for i, abnormal_traj in enumerate(self.abnormal_speed_trajectories):
                    f.write(f"    {i+1}. {abnormal_traj['trajectory_name']}\n")
                    f.write(f"       路径: {abnormal_traj['file_path']}\n")
                    f.write(f"       异常类型: {', '.join(abnormal_traj['abnormal_types'])}\n")
                    f.write(f"       最大速度: ")
                    max_speed_info = []
                    for speed_type, max_val in abnormal_traj['max_speeds'].items():
                        if speed_type in ['vx', 'vy', 'speed_magnitude']:
                            max_speed_info.append(f"{speed_type}={max_val:.3f}m/s")
                        elif speed_type == 'vyaw':
                            max_speed_info.append(f"{speed_type}={max_val:.3f}rad/s")
                    f.write(", ".join(max_speed_info) + "\n")
                    
                    f.write(f"       异常帧数: ")
                    abnormal_frame_info = []
                    for speed_type, count in abnormal_traj['abnormal_frame_counts'].items():
                        percentage = abnormal_traj['abnormal_frame_percentages'][speed_type]
                        abnormal_frame_info.append(f"{speed_type}={count}帧({percentage:.1f}%)")
                    f.write(", ".join(abnormal_frame_info) + "\n\n")
            else:
                f.write(f"  ✅ 所有轨迹速度均在正常范围内\n")
            
            f.write("\n")
            
            # 🎯 新增：全局异常高速帧统计
            f.write("📊 全局异常高速帧统计:\n")
            global_abnormal_frames = distribution_stats['abnormal_speed_analysis']['global_abnormal_frame_counts']
            global_abnormal_percentages = distribution_stats['abnormal_speed_analysis']['global_abnormal_frame_percentages']
            f.write(f"  VX异常高速帧: {global_abnormal_frames['vx']}帧 ({global_abnormal_percentages['vx']:.2f}%)\n")
            f.write(f"  VY异常高速帧: {global_abnormal_frames['vy']}帧 ({global_abnormal_percentages['vy']:.2f}%)\n")
            f.write(f"  VYAW异常高速帧: {global_abnormal_frames['vyaw']}帧 ({global_abnormal_percentages['vyaw']:.2f}%)\n")
            f.write(f"  速度幅值异常高速帧: {global_abnormal_frames['speed_magnitude']}帧 ({global_abnormal_percentages['speed_magnitude']:.2f}%)\n\n")
            
            # Car pose统计
            f.write("🚗 Car Pose统计:\n")
            for key, stats in global_stats['car_pose'].items():
                f.write(f"  {key.upper()}:\n")
                f.write(f"    最小值: {stats['min']:.6f}\n")
                f.write(f"    最大值: {stats['max']:.6f}\n")
                f.write(f"    平均值: {stats['mean']:.6f}\n")
                f.write(f"    标准差: {stats['std']:.6f}\n")
                f.write(f"    中位数: {stats['median']:.6f}\n")
                f.write(f"    范围: {stats['range']:.6f}\n")
                f.write(f"    样本数: {stats['total_samples']}\n\n")
            
            # Velocity统计
            f.write("🏃 Velocity Decomposed统计:\n")
            for key, stats in global_stats['velocity_decomposed'].items():
                f.write(f"  {key.upper()}:\n")
                f.write(f"    最小值: {stats['min']:.6f}\n")
                f.write(f"    最大值: {stats['max']:.6f}\n")
                f.write(f"    平均值: {stats['mean']:.6f}\n")
                f.write(f"    标准差: {stats['std']:.6f}\n")
                f.write(f"    中位数: {stats['median']:.6f}\n")
                f.write(f"    范围: {stats['range']:.6f}\n")
                f.write(f"    样本数: {stats['total_samples']}\n\n")
            
            # 🎯 新增：数据分布百分比分析
            f.write("📈 数据分布百分比分析:\n")
            f.write("=" * 60 + "\n\n")
            
            # 速度分布百分比
            f.write("🚀 速度分布百分比:\n")
            for key, dist_stats in distribution_stats['velocity_distribution'].items():
                if key == 'speed_magnitude':
                    f.write(f"  SPEED MAGNITUDE (速度幅值):\n")
                    f.write(f"    静止状态 (< 0.01 m/s): {dist_stats['stationary_pct']:.2f}%\n")
                    f.write(f"    缓慢移动 (0.01-0.1 m/s): {dist_stats['slow_pct']:.2f}%\n")
                    f.write(f"    正常速度 (0.1-0.4 m/s): {dist_stats['normal_pct']:.2f}%\n")
                    f.write(f"    快速移动 (0.4-0.5 m/s): {dist_stats['fast_pct']:.2f}%\n")
                    f.write(f"    异常高速 (≥ 0.5 m/s): {dist_stats['abnormal_fast_pct']:.2f}%\n")  # 🎯 新增异常高速
                    f.write(f"    最大速度: {dist_stats['max_speed']:.3f} m/s\n")
                    f.write(f"    平均速度: {dist_stats['avg_speed']:.3f} m/s\n\n")
                else:
                    f.write(f"  {key.upper()}:\n")
                    f.write(f"    接近零 (< 0.01): {dist_stats['near_zero_pct']:.2f}%\n")
                    f.write(f"    低速 (0.01-0.1): {dist_stats['low_speed_pct']:.2f}%\n")
                    f.write(f"    中速 (0.1-0.3): {dist_stats['medium_speed_pct']:.2f}%\n")
                    if key in ['vx', 'vy']:
                        f.write(f"    高速 (0.3-0.5): {dist_stats['high_speed_pct']:.2f}%\n")
                        f.write(f"    异常高速 (≥ 0.5): {dist_stats['abnormal_high_speed_pct']:.2f}%\n")  # 🎯 新增异常高速
                    else:  # vyaw
                        f.write(f"    高速 (0.3-0.6): {dist_stats['high_speed_pct']:.2f}%\n")
                        f.write(f"    异常高速 (≥ 0.6): {dist_stats['abnormal_high_speed_pct']:.2f}%\n")  # 🎯 新增异常高速
                    f.write(f"    正向运动: {dist_stats['positive_pct']:.2f}%\n")
                    f.write(f"    负向运动: {dist_stats['negative_pct']:.2f}%\n")
                    f.write(f"    异常值 (±3σ): {dist_stats['outliers_pct']:.2f}%\n\n")
            
            # 位置分布百分比
            f.write("📍 位置分布百分比:\n")
            for key, dist_stats in distribution_stats['car_pose_distribution'].items():
                if key == 'angle':
                    f.write(f"  ANGLE (角度):\n")
                    f.write(f"    第一象限 (0-90°): {dist_stats['quadrant_1_pct']:.2f}%\n")
                    f.write(f"    第二象限 (90-180°): {dist_stats['quadrant_2_pct']:.2f}%\n")
                    f.write(f"    第三象限 (180-270°): {dist_stats['quadrant_3_pct']:.2f}%\n")
                    f.write(f"    第四象限 (270-360°): {dist_stats['quadrant_4_pct']:.2f}%\n")
                    f.write(f"    角度变化范围: {dist_stats['angle_range_deg']:.1f}° ({dist_stats['angle_range_rad']:.3f} rad)\n")
                    f.write(f"    异常值 (±3σ): {dist_stats['outliers_pct']:.2f}%\n\n")
                else:
                    f.write(f"  {key.upper()}:\n")
                    f.write(f"    接近原点 (< 0.1m): {dist_stats['near_origin_pct']:.2f}%\n")
                    f.write(f"    小范围 (0.1-1.0m): {dist_stats['small_range_pct']:.2f}%\n")
                    f.write(f"    中等范围 (1.0-3.0m): {dist_stats['medium_range_pct']:.2f}%\n")
                    f.write(f"    大范围 (≥ 3.0m): {dist_stats['large_range_pct']:.2f}%\n")
                    f.write(f"    正向位置: {dist_stats['positive_pct']:.2f}%\n")
                    f.write(f"    负向位置: {dist_stats['negative_pct']:.2f}%\n")
                    f.write(f"    异常值 (±3σ): {dist_stats['outliers_pct']:.2f}%\n\n")
            
            f.write("=" * 60 + "\n\n")
            
            # 去掉轨迹级别统计的打印
            # f.write("📈 轨迹级别统计 (前10条轨迹):\n")
            # for i, traj_stat in enumerate(self.trajectory_stats[:10]):
            #     f.write(f"  {i+1}. {traj_stat['trajectory_name']} ({traj_stat['num_frames']} 帧)\n")
        
        print(f"✅ 统计报告已保存:")
        print(f"   JSON格式: {report_path}")
        print(f"   文本格式: {readable_report_path}")
        
        return global_stats
    
    def print_summary(self, global_stats):
        """
        打印统计摘要
        
        Args:
            global_stats: 全局统计字典
        """
        print("\n" + "=" * 80)
        print("📊 数据集分析摘要")
        print("=" * 80)
        
        print(f"📈 总体信息:")
        print(f"   总轨迹数: {global_stats['summary']['total_trajectories']}")
        print(f"   总帧数: {global_stats['summary']['total_frames']}")
        print(f"   平均每轨迹帧数: {global_stats['summary']['avg_frames_per_trajectory']:.1f}")
        
        print(f"\n🚗 Car Pose范围:")
        for key, stats in global_stats['car_pose'].items():
            print(f"   {key.upper()}: [{stats['min']:.3f}, {stats['max']:.3f}] (均值: {stats['mean']:.3f})")
        
        print(f"\n🏃 Velocity Decomposed范围:")
        for key, stats in global_stats['velocity_decomposed'].items():
            print(f"   {key.upper()}: [{stats['min']:.3f}, {stats['max']:.3f}] (均值: {stats['mean']:.3f})")
        
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Dataset Analysis Tool - Analyze processed trajectory data')
    parser.add_argument('--processed_data_dir', default='./processed_data',
                       help='Directory containing processed trajectory data')
    parser.add_argument('--output_dir', default='./analysis_results',
                       help='Output directory for analysis results')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of bins for histograms')
    
    args = parser.parse_args()
    
    print(f"🎯 开始分析处理数据目录: {args.processed_data_dir}")
    print(f"📁 分析结果将保存到: {args.output_dir}")
    
    # 创建分析器
    analyzer = DatasetAnalyzer()
    
    # 分析所有数据
    analyzer.analyze_all_data(args.processed_data_dir)
    
    if analyzer.total_trajectories == 0:
        print("❌ 未找到任何有效的轨迹数据")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成分析结果
    print("\n📊 正在生成分析结果...")
    
    # 1. 生成速度分布直方图
    analyzer.create_velocity_histograms(args.output_dir, bins=args.bins)
    
    # 2. 生成car_pose分析图表
    analyzer.create_car_pose_analysis(args.output_dir)
    
    # 3. 保存统计报告
    global_stats = analyzer.save_statistics_report(args.output_dir)
    
    # 4. 打印摘要
    analyzer.print_summary(global_stats)
    
    print(f"\n🎉 数据分析完成! 结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main() 
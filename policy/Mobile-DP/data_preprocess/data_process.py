#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Data Processing Tool - data_process_mission.py

Features:
1. Traverse given dataset paths
2. Process car_pose data (smoothing/outlier processing, consistent with data_utils)
3. Calculate velocity_decomposed
4. Process head_rotation
5. Save new .json files to specified directory
6. Directory structure consistent with data_analysis.py
7. Save visualization images trajectory.png
"""

import os
import json
import numpy as np
import glob
from tqdm import tqdm
import argparse
import shutil
import re
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
import time

# Set non-interactive mode
matplotlib.use('Agg')  # Use non-interactive backend

# Import data processing functions
from data_utils import (
    process_car_pose_to_base_velocity,
    remove_outliers,
    remove_jumps,
    smooth_data
)


class MissionDataProcessor:
    """
    Mission Data Processor - Simplified Version
    For processing vehicle trajectory data, including car_pose processing, velocity calculation and head_rotation processing
    """
    
    def __init__(self):
        """Initialize processor"""
        # Threshold close to 0, values smaller than this will be treated as 0 (1cm)
        self.ZERO_THRESHOLD = 0.01
        
        # 🎯 新增：轨迹过滤统计
        self.filter_stats = {
            'total_trajectories': 0,
            'filtered_trajectories': 0,
            'processed_trajectories': 0,
            'filtered_trajectory_details': [],  # 记录被过滤的轨迹详细信息
            'abnormal_speed_thresholds': {
                'linear_speed_threshold': 0.5,  # m/s - 合理的最大移动速度
                'angular_speed_threshold': 0.6  # rad/s - 合理的最大转向速度 (≈34.4°/秒)
            }
        }
        
    def find_trajectory_folders(self, base_path):
        """
        Find all trajectory folders containing two @ symbols under the given path
        Supports two cases:
        1. Pass parent directory, find trajectory folders within it
        2. Pass trajectory folder directly
        
        Args:
            base_path: Base path
            
        Returns:
            List of trajectory folders
        """
        # Check if path exists
        if not os.path.exists(base_path):
            print(f"Error: Path does not exist - {base_path}")
            return []
        
        # Check if the passed path itself is a trajectory folder (contains two @ symbols)
        folder_name = os.path.basename(base_path.rstrip('/'))
        if folder_name.count('@') == 2:
            print(f"Detected direct trajectory folder input: {folder_name}")
            return [base_path]
        
        # Otherwise, find all subfolders under the given path
        all_folders = glob.glob(os.path.join(base_path, "*"))
        
        # Filter folders containing two @ symbols
        trajectory_folders = []
        for folder in all_folders:
            if os.path.isdir(folder) and os.path.basename(folder).count('@') == 2:
                trajectory_folders.append(folder)
        
        # Sort folders
        trajectory_folders.sort()
        
        print(f"Found {len(trajectory_folders)} trajectory folders in {base_path}")
        return trajectory_folders

    def get_json_path(self, trajectory_folder):
        """
        Get JSON file path in trajectory folder
        
        Args:
            trajectory_folder: Trajectory folder path
            
        Returns:
            JSON file path
        """
        # Remove trailing slash to ensure correct folder name extraction
        trajectory_folder = trajectory_folder.rstrip('/')
        
        # Trajectory folder name
        folder_name = os.path.basename(trajectory_folder)
        
        # JSON file should have same name as folder
        json_path = os.path.join(trajectory_folder, f"{folder_name}.json")
        
        if os.path.exists(json_path):
            return json_path
        else:
            print(f"Warning: JSON file not found - {json_path}")
            return None

    def create_output_directory(self, input_path, output_base_path):
        """
        Create output directory while maintaining original path structure
        Supports direct trajectory folder input
        
        Args:
            input_path: Input path
            output_base_path: Output base path
            
        Returns:
            Created output directory path
        """
        # Check if it's a directly passed trajectory folder
        folder_name = os.path.basename(input_path.rstrip('/'))
        if folder_name.count('@') == 2:
            # Direct trajectory folder input, need to extract robot ID and task info from path
            path_parts = input_path.rstrip('/').split('/')
            
            # Extract robot ID (10094 or factory10016 format)
            robot_id = "unknown"
            for part in path_parts:
                if part.startswith('10') and len(part) == 5 and part.isdigit():
                    robot_id = part
                    break
                elif part.startswith('factory') and len(part) >= 12:  
                    factory_match = re.search(r'(factory10\d{3})', part)
                    if factory_match:
                        robot_id = factory_match.group(1)  # 保留完整的factory10016格式
                        break
            
            # Extract task info (from parent directory of trajectory folder)
            if len(path_parts) >= 2:
                parent_folder = path_parts[-2]  # Parent directory of trajectory folder
                if parent_folder.startswith('202'):  # Date-starting task directory
                    date_task = parent_folder
                else:
                    date_task = "unknown"
            else:
                date_task = "unknown"
            
            # Create output path
            output_path = os.path.join(output_base_path, robot_id, date_task)
            os.makedirs(output_path, exist_ok=True)
            
            return output_path
        
        # Original logic: extract info from parent directory path
        # Extract robot ID (10081, 10082, etc.) - support both /10081/ and factory10016 formats
        robot_id_match = re.search(r'/(10\d{3})/', input_path)
        if robot_id_match:
            robot_id = robot_id_match.group(1)
        else:
            # Try to extract from factory10016 format - 保留完整的factory格式
            factory_match = re.search(r'/(factory10\d{3})/', input_path)
            robot_id = factory_match.group(1) if factory_match else "unknown"
        
        # Extract date and task info (20250514-day-move-box-todesk)
        date_task_match = re.search(r'/([^/]+)$', input_path)
        date_task = date_task_match.group(1) if date_task_match else "unknown"
        
        # Create output path
        output_path = os.path.join(output_base_path, robot_id, date_task)
        os.makedirs(output_path, exist_ok=True)
        
        return output_path

    def process_json_with_angle_unwrap(self, json_path):
        """
        Process JSON file with angle unwrapping to avoid jumps
        
        Args:
            json_path: JSON file path
            
        Returns:
            Processed data dictionary, returns None if failed
        """
        # Check if file exists
        if not os.path.exists(json_path):
            print(f"Error: JSON file does not exist - {json_path}")
            return None
            
        # Read JSON file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            print(f"Successfully loaded data, total {json_data['total']} records")
            
            # Extract data
            data_points = json_data["data"]
            angle_values_raw = np.array([point["car_pose"][2] for point in data_points])
            
            # Angle unwrapping: expand [-π, π] range to continuous angles, avoiding jumps
            angle_values_unwrapped = np.unwrap(angle_values_raw)
            
            # Check and report unwrapped angle range
            angle_min, angle_max = angle_values_unwrapped.min(), angle_values_unwrapped.max()
            print(f"Angle unwrapping: Original range [{angle_values_raw.min():.3f}, {angle_values_raw.max():.3f}] -> "
                  f"Unwrapped range [{angle_min:.3f}, {angle_max:.3f}] (approx [{angle_min/np.pi:.1f}π, {angle_max/np.pi:.1f}π])")
            
            # Update angle values in JSON data
            for i, unwrapped_angle in enumerate(angle_values_unwrapped):
                json_data["data"][i]["car_pose"][2] = float(unwrapped_angle)
            
            # Build data dictionary for subsequent processing
            data_dict = {
                'x_values': np.array([point["car_pose"][0] for point in data_points]),
                'y_values': np.array([point["car_pose"][1] for point in data_points]),
                'angle_values': angle_values_unwrapped,  # Use unwrapped angles
                'height_values': np.array([point["lifting_mechanism_position"] for point in data_points]),
                'json_data': json_data  # Save updated JSON data
            }
            
            return data_dict
            
        except Exception as e:
            print(f"Error processing JSON file: {e}")
            return None
    
    def integrate_velocity_to_trajectory(self, base_velocity_decomposed, initial_pose, dt=1/20):
        """
        Reconstruct trajectory using body frame velocity integration
        
        Args:
            base_velocity_decomposed: Body frame velocity [vx, vy, vyaw]
            initial_pose: Initial pose [x0, y0, theta0]
            dt: Time step (default 20Hz)
            
        Returns:
            Reconstructed trajectory [x, y, angle]
        """
        n_frames = len(base_velocity_decomposed)
        trajectory = np.zeros((n_frames, 3))
        
        # Set initial pose
        trajectory[0] = initial_pose
        
        # Integrate to reconstruct trajectory
        for i in range(1, n_frames):
            # Previous frame pose
            prev_x, prev_y, prev_theta = trajectory[i-1]
            
            # Current frame body velocity (actually velocity from frame i-1 to frame i)
            vx, vy, vyaw = base_velocity_decomposed[i]
            
            # Displacement in body frame
            dx_body = vx * dt
            dy_body = vy * dt
            dtheta = vyaw * dt
            
            # Key: Use previous frame angle for coordinate transformation (consistent with velocity calculation)
            cos_theta = np.cos(prev_theta)
            sin_theta = np.sin(prev_theta)
            
            # Transform from body frame to global frame
            dx_global = dx_body * cos_theta - dy_body * sin_theta
            dy_global = dx_body * sin_theta + dy_body * cos_theta
            
            # Update pose
            trajectory[i, 0] = prev_x + dx_global
            trajectory[i, 1] = prev_y + dy_global
            trajectory[i, 2] = prev_theta + dtheta
        
        return trajectory

    def enhanced_remove_jumps(self, data, threshold=0.3, max_jump_ratio=0.2):
        """
        Enhanced jump detection and correction method
        
        Args:
            data: Input data array
            threshold: Jump detection threshold, reduced to 0.3
            max_jump_ratio: Maximum allowed jump ratio, reduced to 20%
            
        Returns:
            Corrected data
        """
        if len(data) < 5:
            return data.copy()
            
        result = data.copy()
        
        # Use sliding window for jump detection
        window_size = min(5, len(data) // 10)  # Dynamic window size
        
        for i in range(window_size, len(data) - window_size):
            # Calculate difference between current point and surrounding windows
            before_window = result[i-window_size:i]
            after_window = result[i+1:i+window_size+1]
            current_val = result[i]
            
            # Calculate median of surrounding windows (more robust)
            before_median = np.median(before_window)
            after_median = np.median(after_window)
            expected_val = (before_median + after_median) / 2
            
            # Detect jump
            if abs(current_val - expected_val) > threshold:
                # Use local regression replacement
                x_indices = np.concatenate([
                    np.arange(i-window_size, i),
                    np.arange(i+1, i+window_size+1)
                ])
                y_values = np.concatenate([before_window, after_window])
                
                if len(y_values) > 2:
                    # Use linear interpolation
                    result[i] = np.interp(i, x_indices, y_values)
        
        return result

    def enhanced_smooth_data(self, data, strong_smooth=True, iterations=5):
        """
        Enhanced data smoothing method
        
        Args:
            data: Input data
            strong_smooth: Whether to use strong smoothing
            iterations: Number of smoothing iterations
            
        Returns:
            Smoothed data
        """
        if len(data) < 3:
            return data.copy()
        
        result = data.copy()
        
        # Step 1: Use median filter to remove spikes
        if len(result) >= 5:
            kernel_size = min(9, len(result) // 3)
            if kernel_size % 2 == 0:
                kernel_size += 1
            result = medfilt(result, kernel_size=kernel_size)
        
        # Step 2: Use Gaussian filter for strong smoothing
        if strong_smooth:
            sigma = min(3.0, len(result) / 20)  # Dynamic sigma adjustment
            result = gaussian_filter1d(result, sigma=sigma)
        
        # Step 3: Multiple iterations of Savitzky-Golay filter
        window_length = min(15, len(result) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(3, window_length)
        
        try:
            for _ in range(iterations):
                result = savgol_filter(result, window_length, polyorder=2)
        except:
            # Fallback: additional Gaussian filtering
            result = gaussian_filter1d(result, sigma=2.0)
        
        return result

    def process_car_pose_data(self, data_dict, outlier_threshold=3, jump_threshold=0.3, 
                              smooth_iterations=5, strong_smooth=True):
        """
        Enhanced car_pose data processing method
        
        Args:
            data_dict: Dictionary containing original data
            outlier_threshold: Outlier detection threshold, default is 3
            jump_threshold: Reduced jump detection threshold to 0.3
            smooth_iterations: Increased smoothing iterations to 5
            strong_smooth: Whether to use strong smoothing mode
            
        Returns:
            Processed data dictionary
        """
        # Extract vehicle pose data
        car_pose = np.stack([
            data_dict['x_values'],
            data_dict['y_values'],
            data_dict['angle_values']
        ], axis=1)
        
        # Improved processing pipeline:
        # 1. Rough outlier processing first
        # 2. Then jump detection and correction
        # 3. Finally strong smoothing
        
        processed_car_pose = car_pose.copy()
        
        for i in range(3):  # Process x, y, angle separately
            # Step 1: Outlier processing (using stricter threshold)
            processed_car_pose[:, i] = remove_outliers(
                processed_car_pose[:, i], threshold=outlier_threshold
            )
            
            # Step 2: Enhanced jump detection and correction
            processed_car_pose[:, i] = self.enhanced_remove_jumps(
                processed_car_pose[:, i], threshold=jump_threshold
            )
            
            # Step 3: Enhanced smoothing
            processed_car_pose[:, i] = self.enhanced_smooth_data(
                processed_car_pose[:, i], strong_smooth=strong_smooth, 
                iterations=smooth_iterations
            )
        
        # Use data_utils function to calculate velocity (maintain consistency)
        result = process_car_pose_to_base_velocity(
            car_pose=processed_car_pose,
            outlier_threshold=outlier_threshold,
            jump_threshold=jump_threshold,
            smooth_iterations=1,  # Already preprocessed, no need for heavy processing here
            strong_smooth=False
        )
        
        base_velocity_decomposed = result['base_velocity_decomposed']
        print(f"✅ Successfully processed car_pose data with enhanced filtering")
        
        return {
            'processed_car_pose': processed_car_pose,
            'base_velocity_decomposed': base_velocity_decomposed,
            'original_data': data_dict
        }
    
    def read_diversity_json(self, trajectory_folder):
        """
        Read diversity.json file
        
        Args:
            trajectory_folder: Trajectory folder path
            
        Returns:
            Diversity data and parsed instruction mapping
        """
        diversity_json_path = os.path.join(trajectory_folder, "diversity.json")
        global_instruction = ""
        frame_distribute_map = {}
        
        if os.path.exists(diversity_json_path):
            try:
                with open(diversity_json_path, 'r', encoding='utf-8') as f:
                    diversity_data = json.load(f)
                
                # Get global instruction
                global_instruction = diversity_data.get("instruction", "")
                
                # Parse distribute field
                distribute = diversity_data.get("distribute", {})
                
                if distribute:
                    for frame_range, distribute_value in distribute.items():
                        try:
                            # Parse frame range "start_frame end_frame"
                            start_frame, end_frame = map(int, frame_range.split())
                            
                            # Extract distribute_instruction (take first part split by "|")
                            distribute_instruction = distribute_value.split("|")[0].strip()
                            
                            # Set distribute_instruction for all frames in this range
                            for frame_idx in range(start_frame, end_frame + 1):
                                frame_distribute_map[frame_idx] = distribute_instruction
                                
                        except Exception as e:
                            print(f"❌ Failed to parse distribute entry: {frame_range} -> {distribute_value}, error: {e}")
                
                print(f"✅ Successfully read diversity.json, global instruction: '{global_instruction}'")
                
            except Exception as e:
                print(f"❌ Failed to read diversity.json: {e}")
        else:
            print(f"⚠️  diversity.json file not found: {diversity_json_path}")
        
        return global_instruction, frame_distribute_map
    
    def save_processed_json(self, processed_data, original_json_path, output_dir, trajectory_folder):
        """
        Save processed JSON file
        
        Args:
            processed_data: Processed data
            original_json_path: Original JSON file path
            output_dir: Output directory
            trajectory_folder: Trajectory folder path
            
        Returns:
            Saved file path or None if failed
        """
        try:
            # Get original JSON data
            json_data = processed_data['original_data']['json_data']
            total_frames = json_data['total']
            
            # Read diversity.json file
            global_instruction, frame_distribute_map = self.read_diversity_json(trajectory_folder)
            
            # Extract processed data
            base_velocity_decomposed = processed_data['base_velocity_decomposed']
            processed_car_pose = processed_data['processed_car_pose']
            height_values = processed_data['original_data']['height_values']
            
            # Update JSON data
            for i in range(total_frames):
                # Vehicle pose data (keep 2 decimal places)
                x_rounded = round(float(processed_car_pose[i, 0]), 2)
                y_rounded = round(float(processed_car_pose[i, 1]), 2)
                angle_rounded = round(float(processed_car_pose[i, 2]), 2)
                height_rounded = round(float(height_values[i]), 3)  # Height keeps 3 decimal places
                
                # Velocity data (keep 3 decimal places)
                vx_rounded = round(float(base_velocity_decomposed[i, 0]), 3)
                vy_rounded = round(float(base_velocity_decomposed[i, 1]), 3)
                vyaw_rounded = round(float(base_velocity_decomposed[i, 2]), 3)
                
                # Handle values close to 0
                if abs(x_rounded) < self.ZERO_THRESHOLD or x_rounded == -0.0:
                    x_rounded = 0.0
                if abs(y_rounded) < self.ZERO_THRESHOLD or y_rounded == -0.0:
                    y_rounded = 0.0
                if abs(angle_rounded) < self.ZERO_THRESHOLD or angle_rounded == -0.0:
                    angle_rounded = 0.0
                if abs(height_rounded) < self.ZERO_THRESHOLD or height_rounded == -0.0:
                    height_rounded = 0.0
                if abs(vx_rounded) < self.ZERO_THRESHOLD or vx_rounded == -0.0:
                    vx_rounded = 0.0
                if abs(vy_rounded) < self.ZERO_THRESHOLD or vy_rounded == -0.0:
                    vy_rounded = 0.0
                if abs(vyaw_rounded) < self.ZERO_THRESHOLD or vyaw_rounded == -0.0:
                    vyaw_rounded = 0.0
                
                # Update JSON data
                json_data["data"][i]["car_pose"][0] = x_rounded
                json_data["data"][i]["car_pose"][1] = y_rounded
                json_data["data"][i]["car_pose"][2] = angle_rounded
                
                # 🎯 Add velocity field
                json_data["data"][i]["velocity_decomposed"] = [vx_rounded, vy_rounded, vyaw_rounded]
                
                json_data["data"][i]["lifting_mechanism_position"] = height_rounded
                
                # 🎯 Process head_rotation field - 修改处理逻辑
                # 原始处理逻辑（已注释）：保留原值并进行舍入处理
                # if "head_rotation" in json_data["data"][i]:
                #     head_rotation = json_data["data"][i]["head_rotation"]
                #     if isinstance(head_rotation, list) and len(head_rotation) >= 2:
                #         # Process two head rotation values
                #         head_rot_0 = round(float(head_rotation[0]), 2)
                #         head_rot_1 = round(float(head_rotation[1]), 2)
                #         
                #         # Handle values close to 0
                #         if abs(head_rot_0) < self.ZERO_THRESHOLD or head_rot_0 == -0.0:
                #             head_rot_0 = 0.0
                #         if abs(head_rot_1) < self.ZERO_THRESHOLD or head_rot_1 == -0.0:
                #             head_rot_1 = 0.0
                #         
                #         # Update JSON data
                #         json_data["data"][i]["head_rotation"][0] = head_rot_0
                #         json_data["data"][i]["head_rotation"][1] = head_rot_1
                
                # 新的处理逻辑：强制设置为 [0.0, -1.0]
                json_data["data"][i]["head_rotation"] = [0.0, -1.0]
                
                # 🎯 Add instruction related fields
                if global_instruction:
                    json_data["data"][i]["instruction"] = global_instruction
                
                # Add distribute_instruction for current frame
                if i in frame_distribute_map:
                    json_data["data"][i]["distribute_instruction"] = frame_distribute_map[i]
                else:
                    json_data["data"][i]["distribute_instruction"] = ""
            
            # Force first frame car_pose and velocity to 0
            if total_frames > 0:
                json_data["data"][0]["car_pose"][0] = 0.0
                json_data["data"][0]["car_pose"][1] = 0.0
                json_data["data"][0]["car_pose"][2] = 0.0
                json_data["data"][0]["velocity_decomposed"] = [0.0, 0.0, 0.0]
            
            # Generate output file path
            file_name = os.path.basename(original_json_path)
            output_path = os.path.join(output_dir, file_name)
            
            # Save updated JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Processed JSON data saved to: {output_path}")
            print(f"📊 car_pose data rounded to 2 decimal places, height data to 3 decimal places")
            print(f"🎯 head_rotation 强制设置为 [0.0, -1.0] 所有帧")
            print(f"🚀 Added velocity field: velocity_decomposed: [vx, vy, vyaw] (3 decimal places)")
            print(f"📝 Added instruction fields: instruction and distribute_instruction")
            print(f"🎯 First frame car_pose and velocity set to 0, values close to 0 (< 1cm) replaced with 0.0")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving processed JSON data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_trajectory_visualization(self, processed_data, output_dir, trajectory_name):
        """
        Create trajectory visualization image with four subplots:
        Plot 1: Original vs Smoothed trajectory
        Plot 2: Smoothed vs Velocity integrated trajectory  
        Plot 3: Velocity curves (vx, vy, vyaw)
        Plot 4: Car pose curves (x, y, angle) original vs smoothed
        Layout: First row shows Plot 1 and Plot 2, second row shows Plot 3, third row shows Plot 4
        
        Args:
            processed_data: Processed data
            output_dir: Output directory
            trajectory_name: Trajectory name
        """
        try:
            # Extract data
            original_data = processed_data['original_data']
            processed_car_pose = processed_data['processed_car_pose'].copy()  # Make a copy to avoid modifying original
            base_velocity_decomposed = processed_data['base_velocity_decomposed'].copy()
            
            # Apply consistent first frame zeroing strategy (same as JSON saving)
            if len(processed_car_pose) > 0:
                # Store original first frame values for logging
                original_first_frame = processed_car_pose[0].copy()
                print(f"📍 Original first frame: x={original_first_frame[0]:.3f}, y={original_first_frame[1]:.3f}, angle={original_first_frame[2]:.3f}")
                
                # Force first frame to zero (consistent with JSON output)
                processed_car_pose[0] = [0.0, 0.0, 0.0]
                
                # Also prepare original data with first frame zeroing for consistent comparison
                original_x_zeroed = np.array(original_data['x_values']).copy()
                original_y_zeroed = np.array(original_data['y_values']).copy()
                original_angle_zeroed = np.array(original_data['angle_values']).copy()
                
                # Force original first frame to zero as well for comparison
                if len(original_x_zeroed) > 0:
                    original_x_zeroed[0] = 0.0
                    original_y_zeroed[0] = 0.0
                    original_angle_zeroed[0] = 0.0
                
                # Force first frame velocity to zero (consistent with JSON output)
                if len(base_velocity_decomposed) > 0:
                    base_velocity_decomposed[0] = [0.0, 0.0, 0.0]
            else:
                original_x_zeroed = original_data['x_values']
                original_y_zeroed = original_data['y_values']
                original_angle_zeroed = original_data['angle_values']
            
            # Reconstruct trajectory using velocity integration (now starts from origin)
            initial_pose = [0.0, 0.0, 0.0]  # Always start from origin
            integrated_trajectory = self.integrate_velocity_to_trajectory(
                base_velocity_decomposed, initial_pose
            )
            
            # Create figure using GridSpec for layout control (3 rows)
            fig = plt.figure(figsize=(16, 18))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
            
            # Set main title
            fig.suptitle(f'Trajectory Analysis - {trajectory_name}', fontsize=16, fontweight='bold')
            
            # Plot 1: Original vs Smoothed trajectory (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(original_y_zeroed, original_x_zeroed, 'b-', 
                    alpha=0.6, linewidth=2, label='Original Trajectory (First Frame Zeroed)')
            ax1.plot(processed_car_pose[:, 1], processed_car_pose[:, 0], 'r--', 
                    linewidth=2, label='Smoothed Trajectory (First Frame Zeroed)')
            ax1.scatter(0, 0, color='green', s=100, label='Start (0,0)', zorder=5)
            ax1.scatter(processed_car_pose[-1, 1], processed_car_pose[-1, 0], 
                       color='red', s=100, label='End', zorder=5)
            
            # Add direction arrows to smoothed trajectory
            arrow_indices = np.linspace(0, len(processed_car_pose)-1, 10, dtype=int)
            for i in arrow_indices[1:-1]:  # Skip start and end points
                angle = processed_car_pose[i, 2]
                arrow_length = 0.05
                dx = arrow_length * np.cos(angle)
                dy = arrow_length * np.sin(angle)
                ax1.arrow(processed_car_pose[i, 1], processed_car_pose[i, 0], dy, dx, 
                        head_width=0.02, head_length=0.03, 
                        fc='red', ec='red', alpha=0.7)
            
            ax1.set_title('Plot 1: Original vs Smoothed Trajectory (Consistent First Frame Zeroing)', fontweight='bold')
            ax1.set_xlabel('Y Position (m)')
            ax1.set_ylabel('X Position (m)')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            ax1.legend(fontsize=10)
            
            # Plot 2: Smoothed vs Velocity integrated trajectory with angle visualization (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(processed_car_pose[:, 1], processed_car_pose[:, 0], 'r-', 
                    linewidth=2, label='Smoothed Trajectory')
            ax2.plot(integrated_trajectory[:, 1], integrated_trajectory[:, 0], 'g--', 
                    linewidth=2, label='Velocity Integrated Trajectory')
            ax2.scatter(0, 0, color='green', s=100, label='Start (0,0)', zorder=5)
            ax2.scatter(processed_car_pose[-1, 1], processed_car_pose[-1, 0], 
                       color='red', s=100, label='End', zorder=5)
            
            # Add direction arrows to show angle information
            arrow_step = max(1, len(processed_car_pose) // 15)  # Show ~15 arrows
            arrow_indices = np.arange(0, len(processed_car_pose), arrow_step)
            arrow_length = 0.05  # Arrow length in meters
            
            for i in arrow_indices:
                if i < len(processed_car_pose) and i < len(integrated_trajectory):
                    # Smoothed trajectory arrows (red)
                    angle_smooth = processed_car_pose[i, 2]
                    dx_smooth = arrow_length * np.cos(angle_smooth)
                    dy_smooth = arrow_length * np.sin(angle_smooth)
                    ax2.arrow(processed_car_pose[i, 1], processed_car_pose[i, 0], 
                             dy_smooth, dx_smooth, 
                             head_width=0.015, head_length=0.02, 
                             fc='red', ec='red', alpha=0.7, linewidth=1)
                    
                    # Integrated trajectory arrows (green)
                    angle_integrated = integrated_trajectory[i, 2]
                    dx_integrated = arrow_length * np.cos(angle_integrated)
                    dy_integrated = arrow_length * np.sin(angle_integrated)
                    ax2.arrow(integrated_trajectory[i, 1], integrated_trajectory[i, 0], 
                             dy_integrated, dx_integrated, 
                             head_width=0.015, head_length=0.02, 
                             fc='green', ec='green', alpha=0.7, linewidth=1)
            
            # Calculate trajectory errors (position and angle)
            position_error = np.sqrt(np.sum((processed_car_pose[:, :2] - integrated_trajectory[:, :2])**2, axis=1))
            max_pos_error = np.max(position_error)
            mean_pos_error = np.mean(position_error)
            
            # Calculate angle error (considering angle wrapping)
            angle_diff = processed_car_pose[:, 2] - integrated_trajectory[:, 2]
            # Normalize angle difference to [-π, π]
            angle_error = np.abs(((angle_diff + np.pi) % (2 * np.pi)) - np.pi)
            max_angle_error = np.max(angle_error) * 180 / np.pi  # Convert to degrees
            mean_angle_error = np.mean(angle_error) * 180 / np.pi  # Convert to degrees
            
            ax2.set_title(f'Plot 2: Smoothed vs Velocity Integrated Trajectory (with Angle Arrows)\n'
                         f'Position Error - Max: {max_pos_error:.3f}m, Mean: {mean_pos_error:.3f}m\n'
                         f'Angle Error - Max: {max_angle_error:.1f}°, Mean: {mean_angle_error:.1f}°', 
                         fontweight='bold', fontsize=9)
            ax2.set_xlabel('Y Position (m)')
            ax2.set_ylabel('X Position (m)')
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            ax2.legend(fontsize=9)
            
            # Plot 3: Velocity curves (second row)
            ax3 = fig.add_subplot(gs[1, :])
            
            time_indices = np.arange(len(base_velocity_decomposed))
            
            # Plot three velocity components
            ax3.plot(time_indices, base_velocity_decomposed[:, 0], 'b-', 
                    linewidth=2, label='vx (Forward Velocity m/s)')
            ax3.plot(time_indices, base_velocity_decomposed[:, 1], 'g-', 
                    linewidth=2, label='vy (Lateral Velocity m/s)')
            ax3.plot(time_indices, base_velocity_decomposed[:, 2], 'r-', 
                    linewidth=2, label='vyaw (Angular Velocity rad/s)')
            
            # Calculate velocity statistics
            vx_max, vx_min = np.max(base_velocity_decomposed[:, 0]), np.min(base_velocity_decomposed[:, 0])
            vy_max, vy_min = np.max(base_velocity_decomposed[:, 1]), np.min(base_velocity_decomposed[:, 1])
            vyaw_max, vyaw_min = np.max(base_velocity_decomposed[:, 2]), np.min(base_velocity_decomposed[:, 2])
            
            ax3.set_title(f'Plot 3: Body Frame Velocity Decomposition (First Frame Forced to 0)\n'
                         f'vx: [{vx_min:.3f}, {vx_max:.3f}] m/s | '
                         f'vy: [{vy_min:.3f}, {vy_max:.3f}] m/s | '
                         f'vyaw: [{vyaw_min:.3f}, {vyaw_max:.3f}] rad/s', 
                         fontweight='bold', fontsize=10)
            ax3.set_xlabel('Frame Index')
            ax3.set_ylabel('Velocity (m/s or rad/s)')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add velocity statistics text box
            stats_text = (f'Velocity Statistics:\n'
                         f'vx_max: {vx_max:.3f} m/s\n'
                         f'vy_max: {abs(vy_max) if abs(vy_max) > abs(vy_min) else abs(vy_min):.3f} m/s\n'
                         f'vyaw_max: {abs(vyaw_max) if abs(vyaw_max) > abs(vyaw_min) else abs(vyaw_min):.3f} rad/s')
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 4: Car pose curves - original vs smoothed (third row)
            ax4 = fig.add_subplot(gs[2, :])
            
            # Create time indices for plotting
            frame_indices = np.arange(len(original_x_zeroed))
            
            # Plot x position
            ax4_x = ax4
            line1 = ax4_x.plot(frame_indices, original_x_zeroed, 'b-', 
                              alpha=0.6, linewidth=1.5, label='Original X (First Frame Zeroed)')
            line2 = ax4_x.plot(frame_indices, processed_car_pose[:, 0], 'r--', 
                              linewidth=2, label='Smoothed X (First Frame Zeroed)')
            ax4_x.set_ylabel('X Position (m)', color='blue')
            ax4_x.tick_params(axis='y', labelcolor='blue')
            
            # Create second y-axis for y position
            ax4_y = ax4_x.twinx()
            line3 = ax4_y.plot(frame_indices, original_y_zeroed, 'g-', 
                              alpha=0.6, linewidth=1.5, label='Original Y (First Frame Zeroed)')
            line4 = ax4_y.plot(frame_indices, processed_car_pose[:, 1], 'orange', linestyle='--', 
                              linewidth=2, label='Smoothed Y (First Frame Zeroed)')
            ax4_y.set_ylabel('Y Position (m)', color='green')
            ax4_y.tick_params(axis='y', labelcolor='green')
            
            # Create third y-axis for angle
            ax4_angle = ax4_x.twinx()
            ax4_angle.spines['right'].set_position(('outward', 60))
            line5 = ax4_angle.plot(frame_indices, original_angle_zeroed, 'purple', 
                                  alpha=0.6, linewidth=1.5, label='Original Angle (First Frame Zeroed)')
            line6 = ax4_angle.plot(frame_indices, processed_car_pose[:, 2], 'brown', linestyle='--', 
                                  linewidth=2, label='Smoothed Angle (First Frame Zeroed)')
            ax4_angle.set_ylabel('Angle (rad)', color='purple')
            ax4_angle.tick_params(axis='y', labelcolor='purple')
            
            # Set common x-axis properties
            ax4_x.set_xlabel('Frame Index')
            ax4_x.grid(True, alpha=0.3)
            
            # Calculate pose statistics for title
            x_diff_max = np.max(np.abs(original_x_zeroed - processed_car_pose[:, 0]))
            y_diff_max = np.max(np.abs(original_y_zeroed - processed_car_pose[:, 1]))
            angle_diff_max = np.max(np.abs(original_angle_zeroed - processed_car_pose[:, 2]))
            
            ax4_x.set_title(f'Plot 4: Car Pose Curves - Original vs Smoothed (Consistent First Frame Zeroing)\n'
                           f'Max Diff - X: {x_diff_max:.3f}m | Y: {y_diff_max:.3f}m | Angle: {angle_diff_max:.3f}rad', 
                           fontweight='bold', fontsize=10)
            
            # Combine all lines for legend
            all_lines = line1 + line2 + line3 + line4 + line5 + line6
            all_labels = [l.get_label() for l in all_lines]
            ax4_x.legend(all_lines, all_labels, loc='upper right', fontsize=8)
            
            # Save image
            output_path = os.path.join(output_dir, "trajectory.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
            
            print(f"📊 Trajectory visualization saved: {output_path}")
            print(f"📈 Trajectory reconstruction error - Position Max: {max_pos_error:.3f}m, Mean: {mean_pos_error:.3f}m")
            print(f"🔄 Trajectory reconstruction error - Angle Max: {max_angle_error:.1f}°, Mean: {mean_angle_error:.1f}°")
            print(f"📐 Car pose smoothing differences - X: {x_diff_max:.3f}m, Y: {y_diff_max:.3f}m, Angle: {angle_diff_max:.3f}rad")
            print(f"🎯 Consistent first frame zeroing applied to all data for visualization")
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_trajectory_for_abnormal_speed(self, base_velocity_decomposed, 
                                          linear_speed_threshold=0.5, 
                                          angular_speed_threshold=0.6):
        """
        检测轨迹是否包含异常速度
        
        Args:
            base_velocity_decomposed: 速度数据 [vx, vy, vyaw]
            linear_speed_threshold: 线性速度阈值 (m/s)
            angular_speed_threshold: 角速度阈值 (rad/s)
            
        Returns:
            (has_abnormal_speed, abnormal_info)
        """
        try:
            if len(base_velocity_decomposed) == 0:
                return False, {}
            
            vx = base_velocity_decomposed[:, 0]
            vy = base_velocity_decomposed[:, 1]
            vyaw = base_velocity_decomposed[:, 2]
            
            # 检测异常的线性速度（vx 或 vy 超过阈值）
            abnormal_vx_indices = np.where(np.abs(vx) >= linear_speed_threshold)[0]
            abnormal_vy_indices = np.where(np.abs(vy) >= linear_speed_threshold)[0]
            abnormal_vyaw_indices = np.where(np.abs(vyaw) >= angular_speed_threshold)[0]
            
            # 判断是否有异常速度
            has_abnormal_vx = len(abnormal_vx_indices) > 0
            has_abnormal_vy = len(abnormal_vy_indices) > 0
            has_abnormal_vyaw = len(abnormal_vyaw_indices) > 0
            
            has_abnormal_speed = has_abnormal_vx or has_abnormal_vy or has_abnormal_vyaw
            
            # 收集异常信息
            abnormal_info = {
                'has_abnormal_speed': has_abnormal_speed,
                'abnormal_vx': {
                    'count': len(abnormal_vx_indices),
                    'max_value': np.max(np.abs(vx)) if len(vx) > 0 else 0,
                    'indices': abnormal_vx_indices.tolist() if len(abnormal_vx_indices) > 0 else []
                },
                'abnormal_vy': {
                    'count': len(abnormal_vy_indices),
                    'max_value': np.max(np.abs(vy)) if len(vy) > 0 else 0,
                    'indices': abnormal_vy_indices.tolist() if len(abnormal_vy_indices) > 0 else []
                },
                'abnormal_vyaw': {
                    'count': len(abnormal_vyaw_indices),
                    'max_value': np.max(np.abs(vyaw)) if len(vyaw) > 0 else 0,
                    'indices': abnormal_vyaw_indices.tolist() if len(abnormal_vyaw_indices) > 0 else []
                },
                'total_frames': len(base_velocity_decomposed),
                'thresholds': {
                    'linear_speed_threshold': linear_speed_threshold,
                    'angular_speed_threshold': angular_speed_threshold
                }
            }
            
            return has_abnormal_speed, abnormal_info
            
        except Exception as e:
            print(f"❌ Error checking abnormal speed: {e}")
            return False, {}
    
    def generate_filter_report(self, output_base_path):
        """
        生成轨迹过滤报告
        
        Args:
            output_base_path: 输出基础路径
            
        Returns:
            报告文件路径
        """
        try:
            # 计算统计信息
            total_count = self.filter_stats['total_trajectories']
            filtered_count = self.filter_stats['filtered_trajectories']
            processed_count = self.filter_stats['processed_trajectories']
            
            filtered_percentage = (filtered_count / total_count * 100) if total_count > 0 else 0
            processed_percentage = (processed_count / total_count * 100) if total_count > 0 else 0
            
            # 构建报告数据
            report_data = {
                'filter_summary': {
                    'total_trajectories': total_count,
                    'filtered_trajectories': filtered_count,
                    'processed_trajectories': processed_count,
                    'filtered_percentage': round(filtered_percentage, 2),
                    'processed_percentage': round(processed_percentage, 2)
                },
                'filter_criteria': self.filter_stats['abnormal_speed_thresholds'],
                'filtered_trajectory_details': self.filter_stats['filtered_trajectory_details'],
                'generation_time': {
                    'timestamp': int(time.time()),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                }
            }
            
            # 保存报告文件
            report_path = os.path.join(output_base_path, 'filter_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # 生成可读的文本报告
            text_report_path = os.path.join(output_base_path, 'filter_report.txt')
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TRAJECTORY FILTERING REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("📊 Filtering Summary:\n")
                f.write(f"  Total trajectories found: {total_count}\n")
                f.write(f"  Filtered out (abnormal speed): {filtered_count} ({filtered_percentage:.2f}%)\n")
                f.write(f"  Successfully processed: {processed_count} ({processed_percentage:.2f}%)\n\n")
                
                f.write("🔧 Filter Criteria:\n")
                f.write(f"  Linear speed threshold: {self.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold']:.1f} m/s\n")
                f.write(f"  Angular speed threshold: {self.filter_stats['abnormal_speed_thresholds']['angular_speed_threshold']:.1f} rad/s\n\n")
                
                if filtered_count > 0:
                    f.write("🚫 Filtered Trajectories (Abnormal Speed Detected):\n")
                    for i, filtered_traj in enumerate(self.filter_stats['filtered_trajectory_details'], 1):
                        f.write(f"  {i:2d}. {filtered_traj['trajectory_name']}\n")
                        f.write(f"      Path: {filtered_traj['trajectory_path']}\n")
                        f.write(f"      Reason: ")
                        reasons = []
                        if filtered_traj['abnormal_info']['abnormal_vx']['count'] > 0:
                            reasons.append(f"vx exceeds {self.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold']:.1f} m/s ({filtered_traj['abnormal_info']['abnormal_vx']['count']} frames, max: {filtered_traj['abnormal_info']['abnormal_vx']['max_value']:.3f} m/s)")
                        if filtered_traj['abnormal_info']['abnormal_vy']['count'] > 0:
                            reasons.append(f"vy exceeds {self.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold']:.1f} m/s ({filtered_traj['abnormal_info']['abnormal_vy']['count']} frames, max: {filtered_traj['abnormal_info']['abnormal_vy']['max_value']:.3f} m/s)")
                        if filtered_traj['abnormal_info']['abnormal_vyaw']['count'] > 0:
                            reasons.append(f"vyaw exceeds {self.filter_stats['abnormal_speed_thresholds']['angular_speed_threshold']:.1f} rad/s ({filtered_traj['abnormal_info']['abnormal_vyaw']['count']} frames, max: {filtered_traj['abnormal_info']['abnormal_vyaw']['max_value']:.3f} rad/s)")
                        f.write("; ".join(reasons) + "\n")
                        f.write(f"      Total frames: {filtered_traj['abnormal_info']['total_frames']}\n\n")
                else:
                    f.write("✅ No trajectories were filtered out - all trajectories passed the speed criteria.\n\n")
                
                f.write(f"📅 Report generated: {report_data['generation_time']['datetime']}\n")
            
            print(f"📋 Filter report saved:")
            print(f"   JSON format: {report_path}")
            print(f"   Text format: {text_report_path}")
            print(f"📊 Filtering Statistics:")
            print(f"   Total: {total_count}, Filtered: {filtered_count} ({filtered_percentage:.1f}%), Processed: {processed_count} ({processed_percentage:.1f}%)")
            
            return report_path
            
        except Exception as e:
            print(f"❌ Error generating filter report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_symlinks_for_trajectory(self, trajectory_folder, trajectory_output_path):
        """
        为轨迹创建软连接，连接视频文件和diversity.json
        
        Args:
            trajectory_folder: 原始轨迹文件夹路径
            trajectory_output_path: 目标轨迹文件夹路径
            
        Returns:
            创建的软连接数量
        """
        created_links = 0
        
        # 需要连接的文件列表
        files_to_link = [
            'leftImg.mp4',
            'rightImg.mp4', 
            'faceImg.mp4',
            'diversity.json'
        ]
        
        for file_name in files_to_link:
            source_file = os.path.join(trajectory_folder, file_name)
            target_link = os.path.join(trajectory_output_path, file_name)
            
            # 检查源文件是否存在
            if os.path.exists(source_file):
                try:
                    # 如果目标链接已存在，先删除
                    if os.path.lexists(target_link):
                        os.unlink(target_link)
                    
                    # 创建软连接
                    os.symlink(source_file, target_link)
                    created_links += 1
                    print(f"📎 Created symlink: {file_name}")
                    
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create symlink for {file_name}: {e}")
            else:
                print(f"📄 File not found (skipping): {file_name}")
        
        return created_links
    
    def create_symlinks_for_dataset(self, dataset_source_path, dataset_output_path):
        """
        为数据集创建软连接，连接.pdf、record文件夹和report.json
        
        Args:
            dataset_source_path: 原始数据集路径
            dataset_output_path: 目标数据集路径
            
        Returns:
            创建的软连接数量
        """
        created_links = 0
        
        try:
            # 查找.pdf文件
            pdf_files = glob.glob(os.path.join(dataset_source_path, "*.pdf"))
            for pdf_file in pdf_files:
                pdf_name = os.path.basename(pdf_file)
                target_link = os.path.join(dataset_output_path, pdf_name)
                
                try:
                    if os.path.lexists(target_link):
                        os.unlink(target_link)
                    os.symlink(pdf_file, target_link)
                    created_links += 1
                    print(f"📎 Created symlink for PDF: {pdf_name}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create symlink for {pdf_name}: {e}")
            
            # 连接record文件夹
            record_folder = os.path.join(dataset_source_path, "record")
            if os.path.exists(record_folder) and os.path.isdir(record_folder):
                target_link = os.path.join(dataset_output_path, "record")
                try:
                    if os.path.lexists(target_link):
                        os.unlink(target_link)
                    os.symlink(record_folder, target_link)
                    created_links += 1
                    print(f"📎 Created symlink for record folder")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create symlink for record folder: {e}")
            
            # 连接report.json文件
            report_file = os.path.join(dataset_source_path, "report.json")
            if os.path.exists(report_file):
                target_link = os.path.join(dataset_output_path, "report.json")
                try:
                    if os.path.lexists(target_link):
                        os.unlink(target_link)
                    os.symlink(report_file, target_link)
                    created_links += 1
                    print(f"📎 Created symlink for report.json")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to create symlink for report.json: {e}")
            
        except Exception as e:
            print(f"❌ Error creating dataset symlinks: {e}")
        
        return created_links
    
    def process_single_trajectory(self, json_path, trajectory_folder, trajectory_output_path,
                                  outlier_threshold=3, jump_threshold=0.3, 
                                  smooth_iterations=3, strong_smooth=True,
                                  enable_filtering=False, enable_visualization=False):
        """
        Process single trajectory with optional filtering and visualization
        
        Args:
            json_path: JSON file path
            trajectory_folder: Trajectory folder path
            trajectory_output_path: Trajectory output path
            outlier_threshold: Outlier detection threshold
            jump_threshold: Jump detection threshold
            smooth_iterations: Number of smoothing iterations
            strong_smooth: Whether to use strong smoothing mode
            enable_filtering: Whether to enable abnormal speed filtering
            enable_visualization: Whether to create trajectory visualization
            
        Returns:
            Processing result (success/failure/filtered)
        """
        try:
            trajectory_name = os.path.basename(trajectory_folder)
            print(f"🎯 Processing trajectory: {trajectory_name}")
            
            # 更新总轨迹计数
            self.filter_stats['total_trajectories'] += 1
            
            # Step 1: Angle unwrapping processing
            processed_data_dict = self.process_json_with_angle_unwrap(json_path)
            if not processed_data_dict:
                print(f"❌ Angle unwrapping failed: {trajectory_name}")
                return False
            
            # Step 2: car_pose data processing and velocity calculation
            processed_data = self.process_car_pose_data(
                data_dict=processed_data_dict,
                outlier_threshold=outlier_threshold,
                jump_threshold=jump_threshold,
                smooth_iterations=smooth_iterations,
                strong_smooth=strong_smooth
            )
            
            # 🎯 Step 2.5: 异常速度检测和过滤
            if enable_filtering:
                base_velocity_decomposed = processed_data['base_velocity_decomposed']
                has_abnormal_speed, abnormal_info = self.check_trajectory_for_abnormal_speed(
                    base_velocity_decomposed,
                    self.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold'],
                    self.filter_stats['abnormal_speed_thresholds']['angular_speed_threshold']
                )
                
                if has_abnormal_speed:
                    # 记录过滤的轨迹信息
                    filtered_trajectory_info = {
                        'trajectory_name': trajectory_name,
                        'trajectory_path': trajectory_folder,
                        'json_path': json_path,
                        'abnormal_info': abnormal_info,
                        'filter_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    }
                    self.filter_stats['filtered_trajectory_details'].append(filtered_trajectory_info)
                    self.filter_stats['filtered_trajectories'] += 1
                    
                    print(f"🚫 Trajectory filtered due to abnormal speed: {trajectory_name}")
                    print(f"   vx abnormal frames: {abnormal_info['abnormal_vx']['count']} (max: {abnormal_info['abnormal_vx']['max_value']:.3f} m/s)")
                    print(f"   vy abnormal frames: {abnormal_info['abnormal_vy']['count']} (max: {abnormal_info['abnormal_vy']['max_value']:.3f} m/s)")
                    print(f"   vyaw abnormal frames: {abnormal_info['abnormal_vyaw']['count']} (max: {abnormal_info['abnormal_vyaw']['max_value']:.3f} rad/s)")
                    return "filtered"
                else:
                    print(f"✅ Trajectory passed speed filtering: {trajectory_name}")
            
            # 🎯 Step 2.8: 只有通过过滤的轨迹才创建输出目录
            os.makedirs(trajectory_output_path, exist_ok=True)
            print(f"📁 Created output directory for valid trajectory: {trajectory_output_path}")
            
            # 🎯 Step 3: Create visualization (only if enabled and for trajectories that pass filtering)
            if enable_visualization:
                print(f"📊 Creating trajectory visualization...")
                vis_success = self.create_trajectory_visualization(
                    processed_data=processed_data,
                    output_dir=trajectory_output_path,
                    trajectory_name=trajectory_name
                )
                
                if not vis_success:
                    print(f"⚠️  Visualization creation failed, but continuing processing: {trajectory_name}")
            else:
                print(f"⏩ Skipping visualization (disabled)")
            
            # Step 4: Save processed JSON data (only for trajectories that pass filtering)
            final_json_path = self.save_processed_json(
                processed_data=processed_data,
                original_json_path=json_path,
                output_dir=trajectory_output_path,
                trajectory_folder=trajectory_folder
            )
            
            if final_json_path:
                # 🎯 Step 5: 创建软连接，连接视频文件和diversity.json
                print(f"📎 Creating symlinks for trajectory files...")
                symlink_count = self.create_symlinks_for_trajectory(trajectory_folder, trajectory_output_path)
                print(f"📎 Created {symlink_count} symlinks for trajectory: {trajectory_name}")
                
                self.filter_stats['processed_trajectories'] += 1
                print(f"✅ Successfully completed trajectory processing: {trajectory_name}")
                return True
            else:
                print(f"❌ Failed to save final JSON: {trajectory_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing trajectory: {trajectory_name}, error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_datasets(self, dataset_paths, output_base_path,
                         outlier_threshold=3, jump_threshold=0.3, 
                         smooth_iterations=3, strong_smooth=True,
                         enable_filtering=False, enable_visualization=False):
        """
        Process multiple dataset paths with optional filtering and visualization
        
        Args:
            dataset_paths: List of dataset paths
            output_base_path: Output base path
            outlier_threshold: Outlier detection threshold
            jump_threshold: Jump detection threshold
            smooth_iterations: Number of smoothing iterations
            strong_smooth: Whether to use strong smoothing mode
            enable_filtering: Whether to enable abnormal speed filtering
            enable_visualization: Whether to create trajectory visualizations
            
        Returns:
            Processing result statistics
        """
        print(f"🔍 Starting data processing, processing {len(dataset_paths)} dataset paths")
        if enable_filtering:
            print(f"🚫 Filtering enabled - trajectories with abnormal speed will be filtered out:")
            print(f"   Linear speed threshold: {self.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold']:.1f} m/s")
            print(f"   Angular speed threshold: {self.filter_stats['abnormal_speed_thresholds']['angular_speed_threshold']:.1f} rad/s")
        
        if enable_visualization:
            print(f"📊 Visualization enabled - trajectory.png will be saved for each trajectory")
        else:
            print(f"⏩ Visualization disabled - no trajectory.png will be created (faster processing)")
        
        # Create output base directory
        os.makedirs(output_base_path, exist_ok=True)
        
        # Statistics
        total_processed = 0
        total_failed = 0
        total_filtered = 0
        processed_datasets = {}
        
        # Process each dataset path
        for base_path in dataset_paths:
            print(f"\n📂 Processing dataset path: {base_path}")
            
            # Find trajectory folders
            trajectory_folders = self.find_trajectory_folders(base_path)
            
            # Create corresponding output directory
            output_path = self.create_output_directory(base_path, output_base_path)
            print(f"📁 Output directory: {output_path}")
            
            dataset_processed = 0
            dataset_failed = 0
            dataset_filtered = 0
            
            # Process each trajectory
            for trajectory_folder in tqdm(trajectory_folders, desc="Processing trajectories"):
                # Get trajectory name
                trajectory_name = os.path.basename(trajectory_folder)
                
                # 🎯 先不创建轨迹输出目录，等确认通过过滤后再创建
                trajectory_output_path = os.path.join(output_path, trajectory_name)
                
                # Get JSON file path
                json_path = self.get_json_path(trajectory_folder)
                if not json_path:
                    dataset_failed += 1
                    continue
                
                # Process single trajectory
                result = self.process_single_trajectory(
                    json_path=json_path,
                    trajectory_folder=trajectory_folder,
                    trajectory_output_path=trajectory_output_path,
                    outlier_threshold=outlier_threshold,
                    jump_threshold=jump_threshold,
                    smooth_iterations=smooth_iterations,
                    strong_smooth=strong_smooth,
                    enable_filtering=enable_filtering,
                    enable_visualization=enable_visualization
                )
                
                if result == True:
                    dataset_processed += 1
                elif result == "filtered":
                    dataset_filtered += 1
                    # 🎯 如果轨迹被过滤，确保删除可能创建的空目录
                    if os.path.exists(trajectory_output_path) and not os.listdir(trajectory_output_path):
                        try:
                            os.rmdir(trajectory_output_path)
                            print(f"🗑️  Removed empty directory for filtered trajectory: {trajectory_output_path}")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not remove empty directory {trajectory_output_path}: {e}")
                else:
                    dataset_failed += 1
            
            # 🎯 处理完数据集后，创建数据集级别的软连接
            if dataset_processed > 0:  # 只有当有成功处理的轨迹时才创建数据集软连接
                print(f"📎 Creating dataset-level symlinks...")
                dataset_symlink_count = self.create_symlinks_for_dataset(base_path, output_path)
                print(f"📎 Created {dataset_symlink_count} dataset-level symlinks")
            
            # Record processing results for this dataset
            processed_datasets[base_path] = {
                'processed': dataset_processed,
                'failed': dataset_failed,
                'filtered': dataset_filtered,
                'total': len(trajectory_folders)
            }
            
            total_processed += dataset_processed
            total_failed += dataset_failed
            total_filtered += dataset_filtered
        
        # 🎯 Generate filter report
        if enable_filtering:
            report_path = self.generate_filter_report(output_base_path)
        else:
            report_path = None
        
        # Print statistics
        print("\n" + "="*60)
        print("🎉 Data processing completed!")
        print("="*60)
        print(f"📊 Overall statistics:")
        print(f"  ✅ Successfully processed trajectories: {total_processed}")
        print(f"  🚫 Filtered trajectories (abnormal speed): {total_filtered}")
        print(f"  ❌ Failed trajectories: {total_failed}")
        total_found = total_processed + total_failed + total_filtered
        if total_found > 0:
            print(f"  📈 Processing rate: {(total_processed/total_found*100):.1f}%")
            if enable_filtering:
                print(f"  📈 Filtering rate: {(total_filtered/total_found*100):.1f}%")
        
        if enable_visualization:
            print(f"  📊 Visualization: Enabled - trajectory.png saved for each processed trajectory")
        else:
            print(f"  ⏩ Visualization: Disabled - no images created (faster processing)")
        
        print(f"\n📋 Dataset details:")
        for dataset_path, stats in processed_datasets.items():
            # 从路径中提取设备号和任务名称
            dataset_name = os.path.basename(dataset_path)
            
            # 尝试从路径中提取设备号 - 支持 /10081/ 和 factory10032 格式
            robot_id_match = re.search(r'/(10\d{3})/', dataset_path)
            if robot_id_match:
                robot_id = robot_id_match.group(1)
            else:
                # Try to extract from factory10032 format - 保留完整的factory格式
                factory_match = re.search(r'/(factory10\d{3})/', dataset_path)
                robot_id = factory_match.group(1) if factory_match else "unknown"
            
            # 组合显示格式：设备号-任务名称
            display_name = f"{robot_id}-{dataset_name}"
            
            print(f"  📁 {display_name}:")
            print(f"    ✅ Success: {stats['processed']}/{stats['total']}")
            if enable_filtering:
                print(f"    🚫 Filtered: {stats['filtered']}/{stats['total']}")
            print(f"    ❌ Failed: {stats['failed']}/{stats['total']}")
            if stats['total'] > 0:
                success_rate = (stats['processed']/stats['total']*100)
                print(f"    📈 Success rate: {success_rate:.1f}%")
        
        return {
            'total_processed': total_processed,
            'total_failed': total_failed,
            'total_filtered': total_filtered,
            'datasets': processed_datasets,
            'output_path': output_base_path,
            'filter_report_path': report_path
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simplified Data Processing Tool - Process car_pose data and calculate velocity')
    parser.add_argument('--dataset_paths', nargs='*', 
                       default=[],
                       help='Dataset path list, can be parent directories or trajectory folders directly')
    parser.add_argument('--output_dir', default='./processed_data_filtered',
                       help='Output directory for saving processed JSON files')
    parser.add_argument('--outlier_threshold', type=float, default=3,
                       help='Outlier detection threshold, default is 3')
    parser.add_argument('--jump_threshold', type=float, default=0.3,
                       help='Jump detection threshold, default is 0.3')
    parser.add_argument('--smooth_iterations', type=int, default=3,
                       help='Number of smoothing iterations, default is 3')
    parser.add_argument('--strong_smooth', action='store_true',
                       help='Enable strong smoothing mode')
    # 🎯 过滤相关参数
    parser.add_argument('--enable_filtering', action='store_true',
                       help='Enable abnormal speed filtering (filter out trajectories with abnormal speeds)')
    parser.add_argument('--linear_speed_threshold', type=float, default=0.5,
                       help='Linear speed threshold for filtering (m/s), default is 0.5')
    parser.add_argument('--angular_speed_threshold', type=float, default=0.6,
                       help='Angular speed threshold for filtering (rad/s), default is 0.6')
    # 🎯 新增可视化参数
    parser.add_argument('--visualization', action='store_true',
                       help='Enable trajectory visualization (create trajectory.png for each trajectory)')
    
    args = parser.parse_args()
    
    # Default dataset paths if none provided
    default_dataset_paths = [
        "/x2robot_data/zhengwei/10093/20250712-day-10031-pick-waste-turtle-dual-slave2master",
    ]
    
    # Use default paths if no paths provided
    dataset_paths = args.dataset_paths if args.dataset_paths else default_dataset_paths
    
    print(f"🎯 Processing {len(dataset_paths)} dataset paths:")
    for i, path in enumerate(dataset_paths, 1):
        print(f"  {i:2d}. {path}")
    
    # Create processor
    processor = MissionDataProcessor()
    
    # 🎯 Update threshold settings from command line arguments
    if args.enable_filtering:
        processor.filter_stats['abnormal_speed_thresholds']['linear_speed_threshold'] = args.linear_speed_threshold
        processor.filter_stats['abnormal_speed_thresholds']['angular_speed_threshold'] = args.angular_speed_threshold
        print(f"🔧 Custom thresholds: Linear={args.linear_speed_threshold:.1f} m/s, Angular={args.angular_speed_threshold:.1f} rad/s")
    
    # Process datasets
    result = processor.process_datasets(
        dataset_paths=dataset_paths,
        output_base_path=args.output_dir,
        outlier_threshold=args.outlier_threshold,
        jump_threshold=args.jump_threshold,
        smooth_iterations=args.smooth_iterations,
        strong_smooth=args.strong_smooth,
        enable_filtering=args.enable_filtering,
        enable_visualization=args.visualization
    )
    
    print("🎉 Data processing task completed!")


if __name__ == "__main__":
    # Run with command line arguments
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"🎉 Data processing task completed in {end_time - start_time:.2f} seconds")
    # Example usage (can uncomment for testing)
    """
    # Example parameters
    dataset_paths = [
        "/path/to/your/dataset1",
        "/path/to/your/dataset2"
    ]
    output_dir = "/path/to/output/directory"
    
    # Create processor and run
    processor = MissionDataProcessor()
    result = processor.process_datasets(
        dataset_paths=dataset_paths,
        output_base_path=output_dir,
        outlier_threshold=3,
        jump_threshold=1.0,
        smooth_iterations=3,
        strong_smooth=True
    )
    """
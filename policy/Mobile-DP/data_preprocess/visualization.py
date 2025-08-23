import matplotlib.pyplot as plt
import numpy as np
import os


class VehicleDataVisualizer:
    """
    Vehicle data visualizer for plotting vehicle trajectory data charts.
    """
    
    def __init__(self, output_dir="./vehicle_visualization"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Output directory for visualization
        """
        self.output_dir = output_dir
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Visualization results will be saved to: {os.path.abspath(self.output_dir)}")
    
    def _calculate_y_limits(self, processed_data, margin_percent=0.1):
        """
        Calculate y-axis range based on processed data
        
        Args:
            processed_data: Processed data dictionary
            margin_percent: Margin percentage for y-axis range
            
        Returns:
            Dictionary containing y-axis limits for different data types
        """
        y_limits = {}
        
        if processed_data and 'processed' in processed_data:
            proc_data = processed_data['processed']
            
            # Position limits
            if 'x_values' in proc_data and len(proc_data['x_values']) > 0:
                x_min, x_max = np.min(proc_data['x_values']), np.max(proc_data['x_values'])
                x_range = x_max - x_min
                margin = x_range * margin_percent if x_range > 0 else 0.1
                y_limits['x'] = (x_min - margin, x_max + margin)
            
            if 'y_values' in proc_data and len(proc_data['y_values']) > 0:
                y_min, y_max = np.min(proc_data['y_values']), np.max(proc_data['y_values'])
                y_range = y_max - y_min
                margin = y_range * margin_percent if y_range > 0 else 0.1
                y_limits['y'] = (y_min - margin, y_max + margin)
            
            if 'angle_values' in proc_data and len(proc_data['angle_values']) > 0:
                angle_min, angle_max = np.min(proc_data['angle_values']), np.max(proc_data['angle_values'])
                angle_range = angle_max - angle_min
                margin = angle_range * margin_percent if angle_range > 0 else 0.1
                y_limits['angle'] = (angle_min - margin, angle_max + margin)
            
            if 'height_values' in proc_data and len(proc_data['height_values']) > 0:
                height_min, height_max = np.min(proc_data['height_values']), np.max(proc_data['height_values'])
                height_range = height_max - height_min
                margin = height_range * margin_percent if height_range > 0 else 0.1
                y_limits['height'] = (height_min - margin, height_max + margin)
        
        # Velocity limits (only decomposed velocities)
        if processed_data and 'velocities' in processed_data and 'processed' in processed_data['velocities']:
            vel_data = processed_data['velocities']['processed']
            
            if 'vx' in vel_data and len(vel_data['vx']) > 0:
                vx_min, vx_max = np.min(vel_data['vx']), np.max(vel_data['vx'])
                vx_range = vx_max - vx_min
                margin = vx_range * margin_percent if vx_range > 0 else 0.1
                y_limits['vx'] = (vx_min - margin, vx_max + margin)
            
            if 'vy' in vel_data and len(vel_data['vy']) > 0:
                vy_min, vy_max = np.min(vel_data['vy']), np.max(vel_data['vy'])
                vy_range = vy_max - vy_min
                margin = vy_range * margin_percent if vy_range > 0 else 0.1
                y_limits['vy'] = (vy_min - margin, vy_max + margin)
            
            if 'vyaw' in vel_data and len(vel_data['vyaw']) > 0:
                vyaw_min, vyaw_max = np.min(vel_data['vyaw']), np.max(vel_data['vyaw'])
                vyaw_range = vyaw_max - vyaw_min
                margin = vyaw_range * margin_percent if vyaw_range > 0 else 0.1
                y_limits['vyaw'] = (vyaw_min - margin, vyaw_max + margin)
        
        return y_limits
    
    def plot_all_in_one(self, data_dict, output_dir=None):
        """
        Visualize all data as subplots in one large figure.
        
        Args:
            data_dict: Dictionary containing original and processed data
            output_dir: Output directory for saving plots
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Calculate y-axis limits based on processed data
        y_limits = self._calculate_y_limits(data_dict)
        
        # Create a figure with 2x3 subplots (simplified layout)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Vehicle Data Analysis - Decomposed Velocities', fontsize=16, fontweight='bold')
        
        # Extract data
        original_data = data_dict.get('original', {})
        processed_data = data_dict.get('processed', {})
        velocities = data_dict.get('velocities', {})
        
        # Get time indices
        if processed_data and 'x_values' in processed_data:
            time_indices = np.arange(len(processed_data['x_values']))
        elif original_data and 'x_values' in original_data:
            time_indices = np.arange(len(original_data['x_values']))
        else:
            time_indices = np.arange(100)  # Default fallback
        
        # Plot 1: X Position
        ax = axes[0, 0]
        if original_data and 'x_values' in original_data:
            ax.plot(time_indices[:len(original_data['x_values'])], original_data['x_values'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if processed_data and 'x_values' in processed_data:
            ax.plot(time_indices[:len(processed_data['x_values'])], processed_data['x_values'], 
                   'r-', linewidth=2, label='Processed')
        ax.set_title('X Position (m)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Position (m)')
        if 'x' in y_limits:
            ax.set_ylim(y_limits['x'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Y Position
        ax = axes[0, 1]
        if original_data and 'y_values' in original_data:
            ax.plot(time_indices[:len(original_data['y_values'])], original_data['y_values'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if processed_data and 'y_values' in processed_data:
            ax.plot(time_indices[:len(processed_data['y_values'])], processed_data['y_values'], 
                   'r-', linewidth=2, label='Processed')
        ax.set_title('Y Position (m)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Position (m)')
        if 'y' in y_limits:
            ax.set_ylim(y_limits['y'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Yaw Angle
        ax = axes[0, 2]
        if original_data and 'angle_values' in original_data:
            ax.plot(time_indices[:len(original_data['angle_values'])], original_data['angle_values'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if processed_data and 'angle_values' in processed_data:
            ax.plot(time_indices[:len(processed_data['angle_values'])], processed_data['angle_values'], 
                   'r-', linewidth=2, label='Processed')
        ax.set_title('Yaw Angle (rad)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Angle (rad)')
        if 'angle' in y_limits:
            ax.set_ylim(y_limits['angle'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: vx (Body frame X velocity)
        ax = axes[1, 0]
        if velocities and 'original' in velocities and 'vx' in velocities['original']:
            ax.plot(time_indices[:len(velocities['original']['vx'])], velocities['original']['vx'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if velocities and 'processed' in velocities and 'vx' in velocities['processed']:
            vx_data = velocities['processed']['vx']
            ax.plot(time_indices[:len(vx_data)], vx_data, 'g-', linewidth=2, label='Processed')
        ax.set_title('Forward Velocity - vx (m/s)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Velocity (m/s)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        if 'vx' in y_limits:
            ax.set_ylim(y_limits['vx'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: vy (Body frame Y velocity)
        ax = axes[1, 1]
        if velocities and 'original' in velocities and 'vy' in velocities['original']:
            ax.plot(time_indices[:len(velocities['original']['vy'])], velocities['original']['vy'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if velocities and 'processed' in velocities and 'vy' in velocities['processed']:
            vy_data = velocities['processed']['vy']
            ax.plot(time_indices[:len(vy_data)], vy_data, 'm-', linewidth=2, label='Processed')
        ax.set_title('Sideways Velocity - vy (m/s)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Velocity (m/s)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        if 'vy' in y_limits:
            ax.set_ylim(y_limits['vy'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: vyaw (Angular velocity)
        ax = axes[1, 2]
        if velocities and 'original' in velocities and 'vyaw' in velocities['original']:
            ax.plot(time_indices[:len(velocities['original']['vyaw'])], velocities['original']['vyaw'], 
                   'b-', alpha=0.6, linewidth=1, label='Original')
        if velocities and 'processed' in velocities and 'vyaw' in velocities['processed']:
            vyaw_data = velocities['processed']['vyaw']
            ax.plot(time_indices[:len(vyaw_data)], vyaw_data, 'c-', linewidth=2, label='Processed')
        ax.set_title('Angular Velocity - vyaw (rad/s)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        if 'vyaw' in y_limits:
            ax.set_ylim(y_limits['vyaw'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout(pad=2.0, w_pad=1.2, h_pad=2.5)
        
        # Save the plot
        output_path = os.path.join(output_dir, "vehicle_data_analysis.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Vehicle data analysis plot saved to: {output_path}")
        
        return output_path
    
    def plot_trajectory_only(self, data_dict, output_dir=None):
        """
        仅绘制车辆轨迹图。
        
        参数:
            data_dict: 包含原始和处理后数据的字典
            output_dir: 输出目录，如果为None则使用默认目录
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # 提取数据
        original = data_dict['original']
        processed = data_dict['processed']
        
        plt.figure(figsize=(10, 8))
        
        # 绘制轨迹 - 注意这里横轴为y, 纵轴为x
        plt.plot(original['y_values'], original['x_values'], 'b-', 
                alpha=0.4, linewidth=2, label='Original Trajectory')
        plt.plot(processed['y_values'], processed['x_values'], 'r-', 
                linewidth=3, label='Processed Trajectory')
        plt.scatter(processed['y_values'][0], processed['x_values'][0], 
                   color='green', s=150, label='Start', zorder=5)
        plt.scatter(processed['y_values'][-1], processed['x_values'][-1], 
                   color='red', s=150, label='End', zorder=5)
        
        # 添加方向箭头
        arrow_indices = np.linspace(0, len(processed['x_values'])-1, 15, dtype=int)
        for i in arrow_indices:
            angle = processed['angle_values'][i]
            arrow_length = 0.1
            dx = arrow_length * np.cos(angle)
            dy = arrow_length * np.sin(angle)
            # 注意：dx变为纵轴方向，dy变为横轴方向
            plt.arrow(processed['y_values'][i], processed['x_values'][i], dy, dx, 
                    head_width=0.05, head_length=0.08, 
                    fc='red', ec='red', alpha=0.8)
        
        plt.title('Vehicle Trajectory', fontsize=16, fontweight='bold')
        plt.xlabel('Y Position (m)', fontsize=14, fontweight='bold')
        plt.ylabel('X Position (m)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(fontsize=12)
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "vehicle_trajectory.png"), dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {os.path.join(output_dir, 'vehicle_trajectory.png')}")
        
        plt.show()
    
    def plot_velocities_only(self, data_dict, output_dir=None):
        """
        仅绘制速度相关图表。
        
        参数:
            data_dict: 包含原始和处理后数据的字典
            output_dir: 输出目录，如果为None则使用默认目录
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # 提取速度数据
        velocities = data_dict['velocities']
        
        # 创建2x3的子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行
        # 线速度
        axes[0, 0].plot(velocities['original']['speed'], 'b-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[0, 0].plot(velocities['processed']['speed'], 'g-', 
                       linewidth=2.5, label='Processed')
        axes[0, 0].set_title('Speed Magnitude', fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Speed (m/s)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['speed'])
        axes[0, 0].set_ylim(y_min, y_max)
        
        # 角速度
        axes[0, 1].plot(velocities['original']['vyaw'], 'r-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[0, 1].plot(velocities['processed']['vyaw'], 'b-', 
                       linewidth=2.5, label='Processed')
        axes[0, 1].set_title('Angular Velocity (vyaw)', fontweight='bold')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('vyaw (rad/s)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['vyaw'])
        axes[0, 1].set_ylim(y_min, y_max)
        
        # vx
        axes[0, 2].plot(velocities['original']['vx'], 'b-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[0, 2].plot(velocities['processed']['vx'], 'g-', 
                       linewidth=2.5, label='Processed')
        axes[0, 2].set_title('Forward Velocity (vx)', fontweight='bold')
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('vx (m/s)')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        axes[0, 2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['vx'])
        axes[0, 2].set_ylim(y_min, y_max)
        
        # 第二行
        # vy
        axes[1, 0].plot(velocities['original']['vy'], 'r-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[1, 0].plot(velocities['processed']['vy'], 'b-', 
                       linewidth=2.5, label='Processed')
        axes[1, 0].set_title('Sideways Velocity (vy)', fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('vy (m/s)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['vy'])
        axes[1, 0].set_ylim(y_min, y_max)
        
        # base_velocity 线速度
        axes[1, 1].plot(velocities['original']['base_velocity'][:, 0], 'b-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[1, 1].plot(velocities['processed']['base_velocity'][:, 0], 'g-', 
                       linewidth=2.5, label='Processed')
        axes[1, 1].set_title('Base Linear Velocity', fontweight='bold')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Linear Vel (m/s)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        axes[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['base_velocity'][:, 0])
        axes[1, 1].set_ylim(y_min, y_max)
        
        # base_velocity 角速度
        axes[1, 2].plot(velocities['original']['base_velocity'][:, 1], 'r-', 
                       alpha=0.4, linewidth=1.5, label='Original')
        axes[1, 2].plot(velocities['processed']['base_velocity'][:, 1], 'b-', 
                       linewidth=2.5, label='Processed')
        axes[1, 2].set_title('Base Angular Velocity', fontweight='bold')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Angular Vel (rad/s)')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        axes[1, 2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(velocities['processed']['base_velocity'][:, 1])
        axes[1, 2].set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "vehicle_velocities.png"), dpi=300, bbox_inches='tight')
            print(f"Velocity plots saved to: {os.path.join(output_dir, 'vehicle_velocities.png')}")
        
        plt.show()
    
    def plot_positions_only(self, data_dict, output_dir=None):
        """
        仅绘制位置相关图表。
        
        参数:
            data_dict: 包含原始和处理后数据的字典
            output_dir: 输出目录，如果为None则使用默认目录
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # 提取数据
        original = data_dict['original']
        processed = data_dict['processed']
        
        # 创建2x2的子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # X位置
        axes[0, 0].plot(original['x_values'], 'b-', alpha=0.4, linewidth=1.5, label='Original')
        axes[0, 0].plot(processed['x_values'], 'r-', linewidth=2.5, label='Processed')
        axes[0, 0].set_title('X Position', fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('X Position (m)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['x_values'])
        axes[0, 0].set_ylim(y_min, y_max)
        
        # Y位置
        axes[0, 1].plot(original['y_values'], 'g-', alpha=0.4, linewidth=1.5, label='Original')
        axes[0, 1].plot(processed['y_values'], 'r-', linewidth=2.5, label='Processed')
        axes[0, 1].set_title('Y Position', fontweight='bold')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Y Position (m)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['y_values'])
        axes[0, 1].set_ylim(y_min, y_max)
        
        # 角度
        axes[1, 0].plot(original['angle_values'], 'r-', alpha=0.4, linewidth=1.5, label='Original')
        axes[1, 0].plot(processed['angle_values'], 'b-', linewidth=2.5, label='Processed')
        axes[1, 0].set_title('Angle', fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Angle (rad)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['angle_values'])
        axes[1, 0].set_ylim(y_min, y_max)
        
        # 高度（如果有）
        if original['height_values'] is not None and processed['height_values'] is not None:
            axes[1, 1].plot(original['height_values'], 'm-', alpha=0.4, linewidth=1.5, label='Original')
            axes[1, 1].plot(processed['height_values'], 'r-', linewidth=2.5, label='Processed')
            axes[1, 1].set_title('Lifting Mechanism Height', fontweight='bold')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Height (m)')
            axes[1, 1].grid(True)
            axes[1, 1].legend()
            # 设置基于处理后数据的y轴范围
            y_min, y_max = self._calculate_y_limits(processed['height_values'])
            axes[1, 1].set_ylim(y_min, y_max)
        else:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "vehicle_positions.png"), dpi=300, bbox_inches='tight')
            print(f"Position plots saved to: {os.path.join(output_dir, 'vehicle_positions.png')}")
        
        plt.show()
    
    def plot_comparison(self, data_dict, output_dir=None):
        """
        绘制原始数据与处理后数据的对比图。
        
        参数:
            data_dict: 包含原始和处理后数据的字典
            output_dir: 输出目录，如果为None则使用默认目录
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # 提取数据
        original = data_dict['original']
        processed = data_dict['processed']
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # X位置对比
        axes[0, 0].plot(original['x_values'], 'b-', alpha=0.6, linewidth=2, label='Original')
        axes[0, 0].plot(processed['x_values'], 'r-', linewidth=2, label='Processed')
        axes[0, 0].set_title('X Position Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('X Position (m)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['x_values'])
        axes[0, 0].set_ylim(y_min, y_max)
        
        # Y位置对比
        axes[0, 1].plot(original['y_values'], 'g-', alpha=0.6, linewidth=2, label='Original')
        axes[0, 1].plot(processed['y_values'], 'r-', linewidth=2, label='Processed')
        axes[0, 1].set_title('Y Position Comparison', fontweight='bold')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Y Position (m)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['y_values'])
        axes[0, 1].set_ylim(y_min, y_max)
        
        # 角度对比
        axes[1, 0].plot(original['angle_values'], 'r-', alpha=0.6, linewidth=2, label='Original')
        axes[1, 0].plot(processed['angle_values'], 'b-', linewidth=2, label='Processed')
        axes[1, 0].set_title('Angle Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Angle (rad)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        # 设置基于处理后数据的y轴范围
        y_min, y_max = self._calculate_y_limits(processed['angle_values'])
        axes[1, 0].set_ylim(y_min, y_max)
        
        # 轨迹对比
        axes[1, 1].plot(original['y_values'], original['x_values'], 'b-', 
                       alpha=0.6, linewidth=2, label='Original')
        axes[1, 1].plot(processed['y_values'], processed['x_values'], 'r-', 
                       linewidth=2, label='Processed')
        axes[1, 1].scatter(processed['y_values'][0], processed['x_values'][0], 
                          color='green', s=100, label='Start', zorder=5)
        axes[1, 1].scatter(processed['y_values'][-1], processed['x_values'][-1], 
                          color='red', s=100, label='End', zorder=5)
        axes[1, 1].set_title('Trajectory Comparison', fontweight='bold')
        axes[1, 1].set_xlabel('Y Position (m)')
        axes[1, 1].set_ylabel('X Position (m)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "data_comparison.png"), dpi=300, bbox_inches='tight')
            print(f"Comparison plots saved to: {os.path.join(output_dir, 'data_comparison.png')}")
        
        plt.show() 
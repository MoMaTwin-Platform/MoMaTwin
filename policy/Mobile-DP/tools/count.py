import os
import yaml
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# 读取YAML配置文件
import os
yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diffusion_policy/config/task/pretrain_pick_waste.yaml")

with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

# 获取数据集路径列表
# dataset_paths = [item['path'] for item in config['task']['dataset_path']]
dataset_paths = [item['path'] for item in config['dataset_path']]

# 用于存储每个大类的轨迹数量
category_counts = defaultdict(int)
total_trajectories = 0

# 更新正则表达式，提取完整的类别名称（如factory10018）
category_pattern = re.compile(r'/x2robot/[^/]+/([^/]+)/')

# 遍历每个路径并统计轨迹
for path in dataset_paths:
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        continue
    
    # 从路径中提取完整的类别名称
    match = category_pattern.search(path)
    if match:
        category = match.group(1)  # 提取完整类别名称，如factory10018
    else:
        category = "unknown category"
    
    # 获取路径下的所有子文件夹
    try:
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # 筛选包含@的子文件夹
        trajectory_dirs = [d for d in subdirs if '@' in d]
        
        # 更新计数
        count = len(trajectory_dirs)
        category_counts[category] += count
        total_trajectories += count
        
        print(f"{path}: {count} 条轨迹 (类别: {category})")
    except Exception as e:
        print(f"处理路径时出错 {path}: {e}")

# 打印每个大类的轨迹数量
print("\n各类别轨迹数量统计:")
for category, count in sorted(category_counts.items()):
    percentage = (count / total_trajectories) * 100 if total_trajectories > 0 else 0
    print(f"类别 {category}: {count} 条轨迹 ({percentage:.2f}%)")

print(f"\n总轨迹数: {total_trajectories}")
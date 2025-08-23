
## 快速开始

### 1. 数据预处理 (`data_process.py`)

#### 基本使用方式

```bash
# 使用默认参数处理数据
python data_process.py

# 指定输入数据集路径
python data_process.py --dataset_paths /path/to/dataset1 /path/to/dataset2
# NOTE: 当前为了方便，大家直接在代码中写死自己需要处理的原始数据路径即可

# 指定输出目录
python data_process.py --output_dir ./processed_data
# NOTE：默认放在 ./processed_data
```

#### 高级参数配置

```bash
# 启用速度过滤（过滤异常高速轨迹，极大概率是轨迹中间异常整体跳变）
python data_process.py --enable_velocity_filtering \
    --linear_speed_threshold 0.3 \
    --angular_speed_threshold 0.3

# 调整数据处理参数
python data_process.py \
    --outlier_threshold 2.0 \
    --jump_threshold 0.2 \
    --smooth_iterations 5 \
    --strong_smooth

# 完整参数示例
python data_process.py \
    --dataset_paths /data/robot1 /data/robot2 \
    --output_dir ./processed_data \
    --enable_velocity_filtering \
    --linear_speed_threshold 0.4 \
    --angular_speed_threshold 0.5 \
    --outlier_threshold 2.5 \
    --jump_threshold 0.3 \
    --smooth_iterations 3 \
    --strong_smooth
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_paths` | 内置默认路径 | 输入数据集路径列表 |
| `--output_dir` | `./processed_data_filtered` | 输出目录 |
| `--enable_velocity_filtering` | False | 启用异常速度过滤 |
| `--linear_speed_threshold` | 0.3 | 线性速度过滤阈值 (m/s) |
| `--angular_speed_threshold` | 0.3 | 角速度过滤阈值 (rad/s) |
| `--outlier_threshold` | 3.0 | 异常值检测阈值 (σ倍数) |
| `--jump_threshold` | 0.3 | 跳变检测阈值 (米) |
| `--smooth_iterations` | 3 | 平滑迭代次数 |
| `--strong_smooth` | False | 启用强平滑模式 |

### 2. 数据统计分析 (`data_analysis.py`)

#### 基本使用方式

```bash
# 分析默认处理数据目录
python data_analysis.py

# 指定处理数据目录和输出目录
python data_analysis.py \
    --processed_data_dir ./processed_data \
    --output_dir ./analysis_results

# 调整直方图参数
python data_analysis.py \
    --processed_data_dir ./processed_data \
    --output_dir ./analysis_results \
    --bins 100
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--processed_data_dir` | `./processed_data` | 处理后数据目录 |
| `--output_dir` | `./analysis_results` | 分析结果输出目录 |
| `--bins` | 50 | 直方图bin数量 |

## 详细功能介绍

### 数据预处理功能 (`data_process.py`)

#### 🔧 数据处理流程

1. **角度展开处理**：
   - 使用 `np.unwrap()` 处理角度跳变
   - 将 `[-π, π]` 范围扩展为连续角度
   - 避免角度跳变对后续处理的影响

2. **多层数据清洗**：
   - **跳变检测**：使用滑动窗口检测突然位置跳跃
   - **三层滤波平滑**：
     - 中值滤波去除尖峰噪声
     - 高斯滤波强力平滑（可选）
     - Savitzky-Golay滤波精细调整

3. **速度计算**：
   - 计算本体坐标系下的速度分解：`[vx, vy, vyaw]`
   - `vx`：前进速度，`vy`：侧向速度，`vyaw`：角速度
   - 使用20Hz采样频率计算

4. **异常速度过滤**（可选）：
   - 检测超过阈值的异常高速轨迹
   - 默认阈值：线性速度 0.3 m/s，角速度 0.3 rad/s
   - 生成详细的过滤报告

#### 📊 输出内容

每个处理的轨迹包含：

- **处理后的JSON文件**：
  - 平滑后的 `car_pose` 数据
  - 计算得到的 `velocity_decomposed` 数据
  - 指令信息 `instruction` 和 `distribute_instruction`

- **可视化图片** (`trajectory.png`)：
  - Plot 1：原始轨迹 vs 平滑轨迹
  - Plot 2：平滑轨迹 vs 速度积分重建轨迹（含角度箭头）
  - Plot 3：速度曲线 (vx, vy, vyaw)
  - Plot 4：位姿曲线对比 (x, y, angle)

- **软连接文件**：
  - 视频文件：`leftImg.mp4`, `rightImg.mp4`, `faceImg.mp4`
  - 配置文件：`diversity.json`
  - 数据集级别文件：PDF报告、record文件夹、report.json

- **过滤报告**（启用过滤时）：
  - `filter_report.json`：详细的过滤统计
  - `filter_report.txt`：过滤报告

### 数据统计分析功能 (`data_analysis.py`)

#### 📈 分析功能

1. **全局统计**：
   - Car pose统计：位置 (x, y) 和角度的最大值、最小值、平均值、标准差
   - Velocity统计：速度分量 (vx, vy, vyaw) 的统计信息
   - 轨迹数量和帧数统计

2. **数据分布分析**：
   - **速度分布百分比**：
     - 静止/低速/中速/高速/异常高速的占比
     - 正向/负向运动占比
     - 异常值占比（±3σ）
   - **位置分布百分比**：
     - 不同距离范围的占比
     - 角度象限分布
   - **速度幅值分析**：`speed_magnitude = √(vx² + vy²)`

3. **异常高速检测**：
   - 检测超过 0.5 m/s (线性) 和 0.6 rad/s (角速度) 的异常情况
   - 统计异常轨迹数量和占比
   - 提供详细的异常轨迹信息

#### 📊 输出内容

- **统计报告**：
  - `statistics_report.json`：完整的统计数据（JSON格式）
  - `statistics_report.txt`：统计报告

- **可视化图表**：
  - `velocity_distribution_histograms.png`：速度分布直方图
  - `car_pose_distribution_analysis.png`：位姿分布分析图

# Test 使用文档

## Dataloader 测试套件说明

### 功能特性
- ✅ ​**多维度参数测试**​  
  覆盖多种数据规模/处理策略组合：
  - 总样本量：`[10, 50, 100]`
  - 批次大小：`[1, 3, 5, 8]`
  - 工作进程数：`[1, 2, 4, 8]`
  - 随机延迟范围：`[(0.001,0.003), (0.005,0.015)]`

- 🔍 ​**智能数据验证**
  - ​**指纹哈希校验**：采用SHA256生成张量唯一指纹
  - ​**容错验证机制**：允许批次顺序不一致，但内容必须严格匹配
  - ​**差异分析**：自动定位不匹配批次并展示关键数据特征

- ⚡ ​**性能评估指标**
  - 基准处理耗时 vs 数据加载器耗时
  - 加速比计算（`speedup = 基准时间 / 加载器时间`）
  - 单批次平均处理耗时

## Util 测试套件说明


### 功能特性
- ✅ ​**多数据类型支持**  
  支持列表、张量、字典等多种数据结构的进程间分配验证：
  - 列表类型：基础元素分配策略验证
  - 张量类型：保持数据连续性验证
  - 字典类型：多键值同步分配策略验证

- 🔢 ​**进程分配策略验证**
  - 边界条件测试（样本数<进程数/刚好分配/余数分配）
  - 数据完整性校验（合并后与原数据一致）
  - 类型一致性检查（输出类型与输入类型严格匹配）

- 🧪 ​**多样化测试案例**
  - 样本数：`[1, 8, 9, 10, 15]`
  - 进程数：`[4, 8]`
  - 数据结构：`[list, tensor, dict]`

## 使用方式

运行完整测试套件（显示实时输出）

```bash
pytest -v -s test_dataloader.py test_utils.py
```

运行指定测试套件：
```bash
# 仅运行Util测试
pytest -v -s test_utils.py

# 仅运行dataloader测试 
pytest -v -s test_dataloader.py
```

快速失败模式（遇到第一个错误即停止）
```bash
pytest -v -s -x test_dataloader.py test_utils.py
```

指定dataloader参数组合测试：
```bash
pytest -v -s test_dataloader.py \
    --total_samples 100 \
    --batch_size 8 \
    --num_workers 4 \
    --sleep_range 0.005,0.015
```

# DynamicRobotDataset 测试套件说明

## 功能介绍

以最小的可运行代码运行DynamicRobotDataset，旨在提供一个使用示例。同时作为一个用于测试的基本文件，可以测试在不同数据处理和不同数据集的情况下数据处理的时间，比如平均每秒可以生产多少个Iter的数据，如果遇到训练降速的问题，怀疑Dataset的生产速度跟不上，可以使用该套件进行检查。注意检查时，需要对齐好输入的数据源Yaml。


## 使用方式

以下面的方式快速启动：

```bash
# 单卡测试模式（没有测试过，不建议使用）
python test_dynamic_dataset.py

# 分布式测试 (示例使用8个进程)
accelerate launch --num_processes 8 test_dynamic_dataset.py
```

### 典型输出示例

```text
Epoch 1/1000 开始，共 325 个批次
Training: 100%|█████████| 325/325 [01:23<00:00, 3.89batch/s]

批次抽检 100:
Key: 'action'
  Tensor shape: torch.Size([32, 21, 8])
Key: 'obs'
  Key: 'agent_pos'
    Tensor shape: torch.Size([32, 0, 8])
  Key: 'camera1'
    Tensor shape: torch.Size([32, 3, 128, 128, 3])
Key: 'dataset_name'
  Tensor shape: torch.Size([32, 100])
    Sample 0: x2robot/assembly_task/episode_001
    Sample 1: x2robot/pick_and_place/episode_005

➤ 已处理 100 个 batch，平均速度: 4.21 batch/s
```
import pytest
import torch
import time
import random
import hashlib
from collections import defaultdict
from torch.utils.data import IterableDataset
from x2robot_dataset.dataloader import DynamicDataLoader

# -------------------- 数据验证函数 --------------------

def tensor_fingerprint(tensor):
    """生成张量的唯一指纹（SHA256哈希）"""
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def validate_batches(actual_batches, expected_batches):
    # 构建指纹字典 {指纹: [批次索引列表]}
    expected_fingerprints = defaultdict(list)
    for idx, batch in enumerate(expected_batches):
        # print("[DEBUG] Expect Batch: ", batch)
        expected_fingerprints[tensor_fingerprint(batch)].append(idx)

    actual_fingerprints = defaultdict(list)
    for idx, batch in enumerate(actual_batches):
        # print("[DEBUG] Actual Batch: ", batch)
        actual_fingerprints[tensor_fingerprint(batch)].append(idx)

    # 检查所有指纹是否匹配
    all_keys = set(expected_fingerprints) | set(actual_fingerprints)
    discrepancies = []

    for key in all_keys:
        expected_count = len(expected_fingerprints.get(key, []))
        actual_count = len(actual_fingerprints.get(key, []))
        
        if expected_count != actual_count:
            discrepancies.append({
                'fingerprint': key,
                'expected': expected_count,
                'actual': actual_count,
                'sample_expected': expected_batches[expected_fingerprints[key][0]] if expected_count>0 else None,
                'sample_actual': actual_batches[actual_fingerprints[key][0]] if actual_count>0 else None
            })

    if discrepancies:
        print("\n[ERROR] Batch validation failed. Details:")
        for d in discrepancies:
            print(f"\n=== Mismatch for fingerprint {d['fingerprint']} ===")
            print(f"Expected occurrences: {d['expected']}")
            print(f"Actual occurrences: {d['actual']}")
            
            if d['sample_expected'] is not None:
                print("\nSample Expected Batch:")
                print(f"Shape: {d['sample_expected'].shape}")
                print(f"First 3 elements: {d['sample_expected'].flatten()[:3]}")
                
            if d['sample_actual'] is not None:
                print("\nSample Actual Batch:")
                print(f"Shape: {d['sample_actual'].shape}")
                print(f"First 3 elements: {d['sample_actual'].flatten()[:3]}")
        
        raise AssertionError(f"Found {len(discrepancies)} batch discrepancies")

# -------------------- 测试数据集 --------------------
class DatasetForTest(IterableDataset):
    def __init__(self, total_samples):
        self.total_samples = total_samples

    def __iter__(self):
        for i in range(self.total_samples):
            yield i

# -------------------- 带随机sleep的collate函数工厂 --------------------
class CollateWrapper:
    def __init__(self, sleep_range = (0.001, 0.003)):
        self.sleep_range = sleep_range

    def __call__(self, batch):
        sleep_time = random.uniform(self.sleep_range[0], self.sleep_range[1])
        time.sleep(sleep_time)
        return torch.tensor(batch, dtype=torch.float32)

def get_serializable_collate_fn(sleep_range):
    return CollateWrapper(sleep_range)

# -------------------- 主测试函数 --------------------
@pytest.mark.parametrize("total_samples", [10, 50, 100])          # 总数据量
@pytest.mark.parametrize("batch_size", [1, 3, 5, 8])             # 批大小
@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])        # 工作进程数
@pytest.mark.parametrize("sleep_range", [(0.001, 0.003), (0.005, 0.015)])  # sleep范围
def test_dataloader(total_samples, batch_size, num_workers, sleep_range):
    """测试动态数据加载器的正确性和性能"""
    
    # 生成测试配置信息
    print(f"\n\n[CONFIG] Samples: {total_samples} | Batch: {batch_size} "
          f"| Workers: {num_workers} | Sleep: {sleep_range}")

    # ---------- 生成基准结果 ----------
    base_dataset = DatasetForTest(total_samples)
    collate_fn = get_serializable_collate_fn(sleep_range)
    
    # 手动处理数据集作为基准
    expected_batches = []
    all_data = list(base_dataset)
    n_batches = total_samples // batch_size
    
    base_start = time.time()
    for i in range(n_batches):
        batch = all_data[i*batch_size : (i+1)*batch_size]
        expected_batches.append(collate_fn(batch))
    base_time = time.time() - base_start

    # ---------- 使用DataLoader处理 ----------
    test_dataset = DatasetForTest(total_samples)
    dataloader = DynamicDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=False
    )

    # 收集结果和计时
    dl_start = time.time()
    actual_batches = [b for b in dataloader]
    dl_time = time.time() - dl_start
    
    # ---------- 结果验证 ----------
    # 1. 检查批次数量
    assert len(actual_batches) == n_batches, "Batch count mismatch"
    
    # 2. 检查数据内容
    validate_batches(actual_batches, expected_batches)
    
    # ---------- 性能统计 ----------
    avg_speedup = (base_time / dl_time) if dl_time > 0 else 0
    print(f"[RESULTS] Baseline: {base_time:.4f}s | Dataloader: {dl_time:.4f}s | "
          f"Speedup: {avg_speedup:.2f}x | "
          f"Avg/Batch: {dl_time/max(1, n_batches):.4f}s")

# -------------------- 边界条件测试 --------------------
@pytest.mark.parametrize("empty_config", [
    {'total_samples': 0, 'batch_size': 5},    # 空数据集
    {'total_samples': 4, 'batch_size': 5}     # 小数据集
])
def test_edge_cases(empty_config):
    """测试边界条件"""
    collate_fn = CollateWrapper((0, 0))  # 无sleep
    dataset = DatasetForTest(empty_config['total_samples'])
    dataloader = DynamicDataLoader(
        dataset=dataset,
        batch_size=empty_config['batch_size'],
        collate_fn=collate_fn
    )
    
    assert len(list(dataloader)) == 0, "Should handle empty cases correctly"
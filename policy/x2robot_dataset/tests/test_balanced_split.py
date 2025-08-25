import torch
from contextlib import contextmanager
import pytest
from x2robot_dataset.common.utils import balanced_split_between_processes
from typing import Union, List, Dict, Tuple

def test_balanced_split_between_processes():
    # 测试案例配置
    test_cases = [
        # 修正后的测试案例
        (list(range(9)), 8, [[0,1], [2], [3], [4], [5], [6], [7], [8]]),  # 修改此处
        (list(range(10)), 4, [[0,1,2], [3,4,5], [6,7], [8,9]]),
        (list(range(1)), 8, [[0], [], [], [], [], [], [], []]),
        (list(range(8)), 8, [[0], [1], [2], [3], [4], [5], [6], [7]]),
        (list(range(15)), 8, [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14]]),
        # 张量测试保持分配策略
        (torch.arange(9), 8, [torch.tensor([0,1]), [2], [3], [4], [5], [6], [7], [8]]),
        # 字典测试同步调整
        (
            {'a': [0,1,2,3,4,5,6,7,8], 'b': list(range(9,18))},
            8,
            [
                {'a':[0,1], 'b':[9,10]},  # rank0
                {'a':[2], 'b':[11]},
                {'a':[3], 'b':[12]},
                {'a':[4], 'b':[13]},
                {'a':[5], 'b':[14]},
                {'a':[6], 'b':[15]},
                {'a':[7], 'b':[16]},
                {'a':[8], 'b':[17]}
            ]
        )
    ]

    for input_data, num_procs, expected in test_cases:
        # 类型统一处理
        input_type = type(input_data)
        is_dict = isinstance(input_data, dict)
        
        # 遍历所有进程
        all_split = []
        for rank in range(num_procs):
            with balanced_split_between_processes(
                inputs=input_data,
                num_processes=num_procs,
                process_index=rank,
                apply_padding=False
            ) as split_data:
                # 结果类型校验
                assert isinstance(split_data, input_type), \
                    f"类型不一致: {type(split_data)} vs {input_type}"
                
                # 字典特殊处理
                if is_dict:
                    keys = split_data.keys()
                    assert set(keys) == set(input_data.keys()), "字典键缺失"
                    for k in keys:
                        assert len(split_data[k]) == len(split_data[next(iter(keys))]), "字典值长度不一致"
                all_split.append(split_data)

        # 结果验证
        _assert_split_correctness(input_data, all_split, expected)

def _assert_split_correctness(original, actual_splits, expected):
    """验证切分结果的正确性"""
    # 类型统一转换
    if isinstance(original, dict):
        merged = {k: [] for k in original.keys()}
        for split in actual_splits:
            for k in split.keys():
                merged[k].extend(split[k])
        for k in original.keys():
            assert merged[k] == original[k], f"字典键{k}数据不匹配"
    elif isinstance(original, torch.Tensor):
        merged = torch.cat(actual_splits)
        assert torch.equal(merged, original), "张量数据不一致"
    else:
        # 列表/元组合并
        merged = []
        for split in actual_splits:
            merged.extend(list(split))
        assert merged == list(original), f"列表数据不匹配, 合并结果: {merged}"

        # 验证每个进程的分配是否符合预期
        for proc_split, expected_split in zip(actual_splits, expected):
            if isinstance(expected_split, list):
                assert list(proc_split) == expected_split, \
                    f"分配错误: 进程预期{expected_split} 实际{list(proc_split)}"
            elif isinstance(expected_split, torch.Tensor):
                assert torch.equal(proc_split, expected_split), \
                    f"张量分配错误: 进程预期{expected_split} 实际{proc_split}"

    # 验证无重复/遗漏
    if isinstance(original, (list, tuple, torch.Tensor)):
        original_list = original.tolist() if isinstance(original, torch.Tensor) else list(original)
        merged_list = merged.tolist() if isinstance(merged, torch.Tensor) else merged
        assert sorted(merged_list) == sorted(original_list), "数据存在重复或丢失"

# 运行测试
if __name__ == "__main__":
    pytest.main(["-v", __file__])
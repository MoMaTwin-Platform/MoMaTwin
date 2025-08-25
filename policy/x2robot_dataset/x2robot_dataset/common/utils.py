import torchvision
import torch
from contextlib import contextmanager
from typing import Union, List, Dict, Tuple


def decode_video_torchvision(file_name, keyframes_only=True, backend = 'pyav'):
    '''
    Decode video using torchvision.io.VideoReader
    '''
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(file_name, "video")
    reader.seek(0, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)

    reader.container.close()
    reader = None
    loaded_frames = torch.stack(loaded_frames).numpy()

    return loaded_frames

@contextmanager
def balanced_split_between_processes(
    inputs: Union[List, Tuple, Dict, torch.Tensor],
    num_processes: int,
    process_index: int,
    apply_padding: bool = False
):
    """
    改进的分布式数据切分方法，实现平衡分配
    Args:
        inputs: 输入数据，支持列表/元组/字典/张量
        num_processes: 总进程数 
        process_index: 当前进程编号
        apply_padding: 是否自动填充最后元素（慎用）
    """
    if num_processes == 1:
        yield inputs
        return

    def _calculate_indices(total: int) -> Tuple[int, int]:
        # 核心分配算法
        base_size = total // num_processes
        remainder = total % num_processes
        
        # 前remainder个进程多分配1个样本
        if process_index < remainder:
            start = process_index * (base_size + 1)
            end = start + (base_size + 1)
        else:
            start = remainder * (base_size + 1) + (process_index - remainder) * base_size
            end = start + base_size
        
        # 边界保护
        start = min(start, total)
        end = min(end, total)
        return start, end

    def _split(data, start, end):
        if isinstance(data, (list, tuple)):
            return data[start:end]
        elif isinstance(data, torch.Tensor):
            return data[start:end]
        elif isinstance(data, dict):
            assert all(len(v) == len(next(iter(data.values()))) for v in data.values()), "字典各值长度必须一致"
            return {k: v[start:end] for k, v in data.items()}
        else:
            raise TypeError(f"不支持的输入类型: {type(data)}")

    if isinstance(inputs, dict):
        total = len(next(iter(inputs.values())))
    else:
        total = len(inputs)

    start_idx, end_idx = _calculate_indices(total)
    
    # 处理索引越界情况
    if start_idx >= total:
        split_data = inputs[-1:] if apply_padding else inputs[0:0]  # 返回空数据
    else:
        split_data = _split(inputs, start_idx, end_idx)

    # 自动填充逻辑（非必要不建议启用）
    if apply_padding and (end_idx - start_idx) < (total // num_processes):
        padding_size = (total // num_processes) - (end_idx - start_idx)
        if isinstance(split_data, list):
            split_data += [split_data[-1]] * padding_size
        elif isinstance(split_data, torch.Tensor):
            split_data = torch.cat([split_data, split_data[-1:].repeat(padding_size, 1)])
        elif isinstance(split_data, dict):
            split_data = {k: v + [v[-1]] * padding_size if isinstance(v, list) else torch.cat([v, v[-1:].repeat(padding_size, 1)]) 
                        for k, v in split_data.items()}

    yield split_data

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_tensor_shapes(data, indent=0):
    """
    递归打印嵌套字典中所有 tensor 的 shape。
    支持 dict、list、tuple 等结构，并特别处理 dataset_name key。
    """
    prefix = "  " * indent

    if isinstance(data, dict):
        for k, v in data.items():
            if k == 'dataset_name':
                # 特别处理 dataset_name，直接打印其内容
                print(f"{prefix}Key: '{k}'")
                if isinstance(v, torch.Tensor):
                    print(f"{prefix}  Tensor shape: {v.shape}")
                else:
                    print(f"{prefix}  Value: {v}")
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

def decode_tensor(tensor):
        # 找到第一个零的位置（如果有）
        zero_indices = (tensor == 0).nonzero(as_tuple=True)[0]
        if len(zero_indices) > 0:
            # 截取到第一个零之前的部分
            valid_length = zero_indices[0].item()
            valid_tensor = tensor[:valid_length]
        else:
            # 如果没有零，使用整个张量
            valid_tensor = tensor
        
        # 将每个整数转换为对应的 ASCII 字符
        try:
            decoded = ''.join([chr(c) for c in valid_tensor.tolist()])
        except ValueError as e:
            # 处理非 ASCII 字符的情况
            print(f"警告: 解码时遇到非 ASCII 字符: {e}")
            decoded = ''.join([chr(c) if 0 <= c <= 127 else '?' for c in valid_tensor.tolist()])
        
        return decoded
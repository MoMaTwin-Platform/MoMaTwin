import torch
import json
import copy
import io
import numpy as np

class TensorCodec:
    def encode(self, data):
        """
        将包含嵌套张量的字典编码为可序列化结构。
        Returns:
            dict: 包含三个键的字典（data_dict, structure, tensor_data）
        """
        structure = []
        tensor_dict = {}

        with torch.cuda.nvtx.range("process dict"):
            non_tensor_data = self._process_dict(data, '', structure, tensor_dict)

        # 合并各类型张量并转换为字节
        
        with torch.cuda.nvtx.range("dump"):
            encoded = {
                'data_dict': json.dumps(non_tensor_data),
                'structure': json.dumps(structure)
            }
            
        for dtype in tensor_dict:
            if tensor_dict[dtype]:
                merged = torch.cat(tensor_dict[dtype])
                encoded[f'tensor_data_{dtype}'] = merged
                
        return encoded

    def _process_dict(self, data, parent_key, structure, tensor_dict):
        """迭代式字典处理 + 内存预分配优化"""
        stack = [(data, parent_key)]  # 使用栈代替递归
        non_tensor = {}
        path_stack = [[]]  # 路径组件栈

        
        while stack:
            current_data, current_parent = stack.pop()
            path_components = path_stack.pop()
            
            # 按插入顺序遍历 (Python3.7+ dict特性)
            for key, value in current_data.items():  
                full_path = path_components + [key]
                
                if isinstance(value, dict):
                    # 处理子字典
                    stack.append((value, current_parent))
                    path_stack.append(full_path.copy())
                    sub_dict = non_tensor
                    for p in full_path:
                        sub_dict = sub_dict.setdefault(p, {})
                elif isinstance(value, torch.Tensor):
                    # 张量处理优化
                    dtype_str = str(value.dtype).split('.')[-1]
                    flat_tensor = value.flatten()
                    
                    if dtype_str not in tensor_dict:
                        tensor_dict[dtype_str] = []
                    
                    tensor_dict[dtype_str].append(flat_tensor)
                    structure.append({
                        'path': '.'.join(full_path),
                        'shape': list(value.shape),
                        'dtype': dtype_str
                    })
                else:
                    # 非张量数据延迟校验
                    sub_dict = non_tensor
                    for p in full_path[:-1]:
                        sub_dict = sub_dict.setdefault(p, {})
                    sub_dict[full_path[-1]] = value
        
        return non_tensor

    def decode(self, encoded):
        """将编码后的数据还原为原始字典结构"""
        non_tensor_data = json.loads(encoded['data_dict'])
        structure = json.loads(encoded['structure'])
        
        # 加载各类型张量数据
        tensor_buffers = {}
        for key in encoded:
            if key.startswith('tensor_data_'):
                dtype = key.split('_')[-1]
                tensor_buffers[dtype] = encoded[key]
        
        decoded = copy.deepcopy(non_tensor_data)
        pointer_dict = {dtype: 0 for dtype in tensor_buffers}
        
        for item in structure:
            path = item['path']
            shape = item['shape']
            dtype = item['dtype']
            num_elements = torch.prod(torch.tensor(shape)).item()
            
            # 从对应类型缓冲区提取数据
            if dtype not in tensor_buffers:
                raise ValueError(f"Missing tensor data for dtype: {dtype}")
                
            buffer = tensor_buffers[dtype]
            tensor = buffer[pointer_dict[dtype] : pointer_dict[dtype]+num_elements]
            tensor = tensor.reshape(shape).to(getattr(torch, dtype))
            pointer_dict[dtype] += num_elements
            
            # 重建字典结构
            keys = path.split('.')
            current_dict = decoded
            for i, key in enumerate(keys[:-1]):
                current_dict = current_dict.setdefault(key, {})
                if not isinstance(current_dict, dict):
                    raise ValueError(f"Structure conflict at {'.'.join(keys[:i+1])}")
            
            current_dict[keys[-1]] = tensor
        
        # 验证数据完整性
        for dtype in pointer_dict:
            if pointer_dict[dtype] != len(tensor_buffers[dtype]):
                raise ValueError(f"{dtype} tensor data length mismatch")
                
        return decoded
    
    
def create_test_data():
    """创建包含多种数据类型的测试数据"""
    return {
        'actions': {
            'empty_tensor': torch.randint(0, 2, (0,640,480), dtype=torch.uint8),
            'binary_sensor': torch.randint(0, 2, (10,), dtype=torch.uint8),
            'continuous_action': torch.randn(5, dtype=torch.float32),
            'index_labels': torch.tensor([1, 2, 3], dtype=torch.int64)
        },
        'metadata': {
            'timestamps': torch.tensor([1001, 1002], dtype=torch.int64),
            'sensor_data': torch.rand(3, 256, dtype=torch.float32)
        },
        'info': {
            'description': "Multi-dtype test sample",
            'version': 1.2
        },
        'dataset_name': "XC Test"
    }

def test_codec():
    """完整的测试流程"""
    # 初始化编解码器
    codec = TensorCodec()
    
    # 创建测试数据
    original_data = create_test_data()
    print(original_data)
    # 编码测试
    encoded = codec.encode(original_data)
    assert 'tensor_data_uint8' in encoded
    assert 'tensor_data_float32' in encoded
    assert 'tensor_data_int64' in encoded
    assert 'data_dict' in encoded
    assert 'structure' in encoded
    
    # 解码测试
    decoded_data = codec.decode(encoded)
    assert decoded_data['actions']['binary_sensor'].dtype == torch.uint8
    assert decoded_data['actions']['continuous_action'].dtype == torch.float32
    assert decoded_data['metadata']['timestamps'].dtype == torch.int64
    
    
    # 结构完整性验证
    def recursive_compare(orig, decoded):
        if isinstance(orig, dict):
            for k in orig:
                assert k in decoded
                recursive_compare(orig[k], decoded[k])
        elif isinstance(orig, torch.Tensor):
            assert torch.equal(orig, decoded)
        else:
            assert orig == decoded
    print(decoded_data)
    recursive_compare(original_data, decoded_data)
    
    print("All tests passed!")

if __name__ == '__main__':
    test_codec()
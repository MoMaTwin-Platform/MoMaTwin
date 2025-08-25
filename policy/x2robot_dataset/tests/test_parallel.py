import torch
from torch.utils.data import IterableDataset, DataLoader
from accelerate import Accelerator

class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

        self.indices = list(range(100))

    def __iter__(self):
        #rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        #world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        for i in self.indices:
            yield i
    
    def __len__(self):
        return len(self.data)
    
    

# 初始化数据
data = range(103)  # 103 个样本
dataset = MyIterableDataset(data)

# 使用 accelerate
accelerator = Accelerator()

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
dataloader = accelerator.prepare(dataloader)

# 验证不同进程的数据
total_samples = 0
for batch in dataloader:
    print(f"Process {accelerator.process_index}: {batch}")
    total_samples += len(batch)

print(f"Process {accelerator.process_index} processed {total_samples} samples.")

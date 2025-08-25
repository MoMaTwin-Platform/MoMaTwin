import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from x2robot_dataset.data_buffer import BufferedDatasetWapper, SlidingWindowSampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import tqdm
class NumDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        x = self.data[index]
        
        return x
    
    def __len__(self):
        return len(self.data)

import numpy as np
import torch
data = []
for i in range(8):
    data.append(NumDataset(torch.from_numpy(np.arange(i*20, (i+1)*20-i))))
dataset = ConcatDataset(data)
dataset = BufferedDatasetWapper(dataset.datasets, buffer_size=4, seed=89)
sampler = SlidingWindowSampler(dataset)

dataloader = DataLoader(dataset, batch_size=10, sampler=sampler, shuffle=False, num_workers=5)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=5)



for epoch in range(10):
    print('epoch: ', epoch)
    fetched_indices = []
    dataset.reset_data(epoch)
    for batch in tqdm.tqdm(dataloader):
        fetched_indices.append(batch)
        print(batch)
    fetched_indices = torch.cat(fetched_indices, dim=0)
    print(fetched_indices.sort()[0])
    print(fetched_indices.shape)
    print([len(dataset.datasets[i].data) for i in range(len(dataset.datasets))])

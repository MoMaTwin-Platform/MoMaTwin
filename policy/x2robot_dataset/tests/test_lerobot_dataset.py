from x2robot_dataset.lerobot_dataset import (
    make_lerobot_dataset,
    LeRDataConfig
)


from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from queue import Queue
from torch.utils.data import get_worker_info
import torch


from torch.utils.data import DataLoader
import tqdm
if __name__ == '__main__':
    data_folders = [\
        '/x2robot/zhengwei/10001/20240617-hang-clothes', \
        '/x2robot/zhengwei/10001/20240618-hang-clothes-rlhf', \
        '/x2robot/zhengwei/10001/20240620-hang-clothes-rlhf'\
    ]
    default_data_config = LeRDataConfig().as_dict()
    train_dataset = make_lerobot_dataset(data_folders=data_folders,
                                        data_configs=[default_data_config]*3,
                                        force_overwrite=False)

    train_loader = DataLoader(train_dataset, batch_size=32, \
        shuffle=True, num_workers=20, pin_memory=True, prefetch_factor=5)
    accelerator = Accelerator()
    train_loader = accelerator.prepare(train_loader)

    rank = accelerator.process_index
    cc = 0
    for data in tqdm.tqdm(train_loader):
        batch = data['observations.face_view']
        print(rank, batch.shape)
        cc += 1
        if cc > 100:
            break
        
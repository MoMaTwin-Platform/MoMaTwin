import zarr
import torch.distributed as dist
from zarr.sync import ThreadSynchronizer,ProcessSynchronizer
import inspect

file_store = {}

def load_data(file_path):
    call_stack = inspect.stack()
    for frame in call_stack:
        print(f"Frame:{frame.filename}, Line:{frame.lineno} Function:{frame.function} {file_path}", flush=True)
    rank = dist.get_rank()
    key = file_path
    if key not in file_store:
        print(f'{rank} Loading', file_path)
        store = zarr.ZipStore(file_path, mode='r')
        # store = zarr.DirectoryStore(file_path)
        buffer = zarr.open(store, mode='r') #,synchronizer=ProcessSynchronizer(path='data/example.sync')
        file_store[key] = buffer
    print(f"{rank}, {file_store}", flush=True)


    # def load_data(self):
    #     rank = torch.distributed.get_rank()
    #     key = self.zarr_path
    #     memory_0 = memory_usage()
    #     file_master.load_data(self.zarr_path)
    #     print(file_master.file_store)
    #     memory_1 = memory_usage()
    #     self.replay_buffer = file_master.file_store[key]
    #     memory_2 = memory_usage()
    #     print(f"load path:{self.zarr_path}, Before:{memory_0}, Stored:{memory_1}, Opened:{memory_2}")
    #     self.replay_buffer_meta = {}
    #     for key in self.replay_buffer.meta:
    #         self.replay_buffer_meta[key] = self.replay_buffer.meta[key]
    
    #     self.replay_buffer_data = {}
    #     for key in self.replay_buffer.data:
    #         self.replay_buffer_data[key] = self.replay_buffer.data[key]

"""
测试文件：测试 IterChunkDataset 的数据加载和分布验证

主要功能：
1. 测试数据加载性能和内存使用
   - 使用 Accelerator 进行多进程数据加载
   - 监控内存使用情况和线程数量
   - 测试不同批次大小下的加载性能

2. 数据完整性和分布验证
   - 检查训练集和验证集的数据分布
   - 验证不同 rank 之间的数据是否有重叠
   - 分析每个 UID 的帧序列完整性
   - 检测训练集和验证集之间的 UID 重叠情况

3. 数据配置测试
   - 支持多种数据源配置（如 x2、hf 等类型）
   - 测试数据预加载和缓冲区设置
   - 验证数据切片和批处理的正确性

使用方法：
- 通过 Accelerator 启动多进程运行
- 可配置 batch_size、预加载池大小等参数
- 支持多轮 epoch 测试
- 生成详细的数据分析报告

输出内容：
- 数据加载性能统计
- 内存使用详情
- 数据分布分析报告
- 数据重叠和完整性检查结果
"""

import io
import time
from torch.utils.data import DataLoader
from accelerate import Accelerator

from x2robot_dataset.lazy_dataset import (
    IterChunkDataset,
    X2RDataProcessingConfig,
    X2RDataChunkConfig,
)

from x2robot_dataset.common.collate_fn import collate_wrapper
from tqdm import tqdm
import torch

from x2robot_dataset.common.data_utils import decode_text


######################### 数据分析工具 #########################
import psutil
import threading
import glob, re
import jsonlines
def get_detailed_memory_usage():
    """获取更详细的内存使用信息"""
    process = psutil.Process()
    memory_info = process.memory_info()
    # 获取当前线程信息
    current_threads = threading.active_count()
    # thread_names = [t.name for t in threading.enumerate()]
    return {
        'rss': f"{memory_info.rss / 1024 / 1024:.2f}MB ",
        'vms': f"{memory_info.vms / 1024 / 1024:.2f}MB ",
        'threads_count': current_threads,
        # 'threads': thread_names,
    }
def get_detailed_thread():
    current_threads = threading.active_count()
    thread_names = [t.name for t in threading.enumerate()]
    return {
        'threads_count': current_threads,
        'threads': thread_names,
    }

def print_rank0(rank,msg,flush=True):
    if rank==0:
        print(msg,flush=flush)

def analyze_data_distribution(epoch):
    def analyze_single_split(split):
        all_rank_data = {}
        rank_files = glob.glob(f'epoch{epoch}_{split}_rank*.jsonl')
        
        # 收集该split的所有数据
        for file in rank_files:
            rank = int(re.search(r'rank(\d+)', file).group(1))
            with jsonlines.open(file) as reader:
                all_rank_data[rank] = set(reader.read())

        # 1. 检查不同rank间的数据重合
        overlapping = {}
        for rank1, data1 in all_rank_data.items():
            for rank2, data2 in all_rank_data.items():
                if rank1 >= rank2:
                    continue
                intersection = data1 & data2
                if intersection:
                    overlapping[f"rank{rank1}-rank{rank2}"] = intersection

        # 2. 检查每个uid的frame完整性
        uid_frames = {}
        all_uids = set()  # 用于记录该split的所有uid
        for rank_data in all_rank_data.values():
            for item in rank_data:
                uid = "_".join(item.split('_')[:-1])
                frame_info = item.split('_')[-1]
                all_uids.add(uid)
                if uid not in uid_frames:
                    uid_frames[uid] = set()
                uid_frames[uid].add(int(frame_info))

        frame_stats = {}
        for uid, frames in uid_frames.items():
            min_frame = min(frames)
            max_frame = max(frames)
            expected_frames = set(range(min_frame, max_frame + 1))
            missing_frames = expected_frames - frames
            missing_ratio = len(missing_frames) / len(expected_frames)
            frame_stats[uid] = {
                'missing_ratio': missing_ratio,
                'missing_frames': sorted(missing_frames),
                "expected_frames": max_frame,
            }

        return {
            'overlapping': overlapping,
            'frame_stats': frame_stats,
            'all_uids': all_uids
        }

    # 分析train和val数据
    train_results = analyze_single_split('train')
    val_results = analyze_single_split('val')

    # 检查train和val之间的uid重合
    train_uids = train_results['all_uids']
    val_uids = val_results['all_uids']
    overlapping_uids = train_uids & val_uids

    return {
        'train': {
            'overlapping': train_results['overlapping'],
            'frame_stats': train_results['frame_stats'],
            'total_uids': len(train_uids)
        },
        'val': {
            'overlapping': val_results['overlapping'],
            'frame_stats': val_results['frame_stats'],
            'total_uids': len(val_uids)
        },
        'train_val_overlap': {
            'overlapping_uids': overlapping_uids,
            'overlap_count': len(overlapping_uids)
        }
    }

def check_rank2uid_frame(rank2uid_frame,epoch,split):
    for rank, uids in rank2uid_frame.items():
        with jsonlines.open(f'epoch{epoch}_{split}_rank{rank}.jsonl', mode='w') as writer:
            # "10001_20240617-hang-clothes@ARM_TEST_MODE@2024_06_17_17_18_57_81"
            writer.write(uids)

class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(280,8)

    def forward(self,batch):
        x = batch["action"].reshape(8,-1)
        frame = batch["frame"].reshape(8,-1)
        x = x.to(self.linear.weight.dtype)
        frame = frame.to(self.linear.weight.dtype)
        y = self.linear(x)
        loss = (frame-y).mean()
        return loss
        
from accelerate import DataLoaderConfiguration
######################### 加载数据&loop #########################
if __name__ == '__main__':
    gradient_accumulation_steps = 8
    batch_size_per_rank = 8
    dataloader_config = DataLoaderConfiguration(split_batches=False, dispatch_batches=False)
    accelerator = Accelerator(dataloader_config=dataloader_config, gradient_accumulation_steps=gradient_accumulation_steps)
    #accelerator = Accelerator(dispatch_batches=False, gradient_accumulation_steps=gradient_accumulation_steps) #split_batches=False,
    #accelerator = Accelerator(dispatch_batches=False, gradient_accumulation_steps=gradient_accumulation_steps) #split_batches=False,
    world_size = accelerator.num_processes
    global_batch_size = batch_size_per_rank * world_size

    # data_folders = [\
    #     '/x2robot/zhengwei/10001/20240617-hang-clothes', \
    #     '/x2robot/zhengwei/10001/20240620-hang-clothes-rlhf', \
    #     'factory10001/20241114-tear-tissue', \
    #     # '/x2robot/Data/x2robot_Open_X_datasets/aloha_mobile/0.zarr.zip', \
    # ]
    data_folders = ['/x2robot/zhengwei/10006/20241225-pick_up-sponge']
    
    data_configs = [X2RDataProcessingConfig(
        train_test_split=0.6,
    ) for _ in range(len(data_folders))]

    data_configs[0] = data_configs[0].update(
        class_type = 'x2',
        train_test_split = 0.9999,
        default_insstruction = 'pick up the sponge',
        action_keys = ['follow_left_ee_cartesian_pos', 'follow_left_ee_rotation', 'follow_left_gripper', 'follow_right_ee_cartesian_pos', 'follow_right_ee_rotation', 'follow_right_gripper'],
    )

    data_configs[1] = data_configs[1].update(
        class_type = 'x2',
        train_test_split = 0.9,
        default_insstruction = 'pick up the fruit on the ground and place in the basket',
        action_keys = ['follow_left_ee_cartesian_pos', 'follow_left_ee_rotation', 'follow_left_gripper', 'follow_right_ee_cartesian_pos', 'follow_right_ee_rotation', 'follow_right_gripper', 'head_actions', 'car_pose'],
    )

    data_configs = [config.as_dict() for config in data_configs]

    data_chunk_config = X2RDataChunkConfig().update(
        right_padding=True,
    )
    # 加载dataset
    train_dataset = IterChunkDataset(
        data_folders,
        data_configs,
        data_chunk_config,
        preload_pool_size = 2,
        num_preloader_threads  = 2,
        max_frame_buffer_size = 100,
        num_frame_producer_threads = 2,
        force_overwrite=True,
        save_meta_data=False,
        split='train',
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        slice_size=batch_size_per_rank,
    )
    total_frames = train_dataset.num_frames
    val_dataset = IterChunkDataset(
        data_folders,
        data_configs,
        data_chunk_config,
        preload_pool_size = 4,
        num_preloader_threads  = 2,
        max_frame_buffer_size = 2000,
        num_frame_producer_threads = 2,
        force_overwrite=False,
        save_meta_data=False,
        split='test',
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        slice_size=batch_size_per_rank,
    )
    total_frames_val = val_dataset.num_frames

    # 设置collate_fn
    collate_fn = collate_wrapper(
            collate_type = 'chunking',
            low_dim_obs_horizon=1,
            img_obs_horizon=1,
            horizon=20,
            action_dim=14,
            is_bi_mode=True,
            sample2instruct=None,
            to_lie_algebra=False,
            sample2imginstruct=None,
            parse_head_action=False,
            mask_type=None,
            mask_keys=None,
            merge_cur_history=False)

    # 加载dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_per_rank, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_per_rank, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    train_dataloader = accelerator.prepare(train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)

    # 计算train/val step
    train_num = int(total_frames // batch_size_per_rank // accelerator.num_processes)
    val_num = int(total_frames_val // batch_size_per_rank // accelerator.num_processes)
    train_loop_break_step = int(train_num//gradient_accumulation_steps)*gradient_accumulation_steps
    # train_loop_break_step = train_num
    print(f"rank {accelerator.process_index} total_frames:{total_frames} train_num {train_num}, train_loop_break_step {train_loop_break_step}, val_num {val_num}",flush=True)
    print(f"rank {accelerator.process_index} batch_size_per_rank {batch_size_per_rank} global_batch_size {global_batch_size}", flush=True)

    fake_model = FakeModel()
    fake_model = accelerator.prepare(fake_model)
    optimizer = torch.optim.Adam(fake_model.parameters(), lr=1e-3)
    optimizer = accelerator.prepare(optimizer)

    for epoch in tqdm(range(2)):
        print(f"EPOCH START | rank {accelerator.process_index} | epoch {epoch} | {get_detailed_thread()}",flush=True)
        st = time.time()
        acum_time = 0
        pbar = tqdm(total=train_num, position=0, leave=False)
        rank2uid_frame_train = {}
        for i, batch in enumerate(train_dataloader):
            # if i>=train_loop_break_step:
            #     # 需要在达到train_num后，手动跳出loop
            #     break
            batch_size = len(batch["uid"])
            pbar.update(1)
            acum_time += time.time()-st
            with accelerator.accumulate(fake_model):
                loss = fake_model(batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            # if i>train_num-20:
            #     print(f"rank {accelerator.process_index} | step {i} | end of step",flush=True)
            # print(f'rank {accelerator.process_index} {[decode_text(item) for item in batch["uid"]]}', flush=True)
            if accelerator.process_index not in rank2uid_frame_train:
                rank2uid_frame_train[accelerator.process_index] = []
            rank2uid_frame_train[accelerator.process_index].extend([decode_text(item)+"_"+str(frame.item()) for item,frame in zip(batch["uid"],batch["frame"])])
            accelerator.wait_for_everyone()

        print(f"TRAIN LOOP END | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        # accelerator.wait_for_everyone()
        # val_dataloader.dataset.dataset.reset_epoch(epoch)
        accelerator.wait_for_everyone()
        print(f"first reset epoch | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)

        pbar = tqdm(total=val_num, position=0, leave=False)
        rank2uid_frame_val = {}
        for i,batch in enumerate(val_dataloader):
            # if i>=val_num:
            #     break
            if i==int(val_num//2):
                print(f"IN VAL LOOP 1 | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
            batch_size = len(batch["uid"])
            pbar.update(1)
            if accelerator.process_index not in rank2uid_frame_val:
                rank2uid_frame_val[accelerator.process_index] = []
            rank2uid_frame_val[accelerator.process_index].extend([decode_text(item)+"_"+str(frame.item()) for item,frame in zip(batch["uid"],batch["frame"])])
            accelerator.wait_for_everyone()

        print(f" VAL LOOP 1 END | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        print("*************************************************************************************")
        accelerator.wait_for_everyone()
        val_dataloader.dataset.dataset.reset_epoch(epoch)
        accelerator.wait_for_everyone()
        print(f"second reset epoch | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        pbar = tqdm(total=val_num, position=0, leave=False)
        rank2uid_frame_val = {}
        for i,batch in enumerate(val_dataloader):
            if i>=val_num:
                break
            if i==int(val_num//2):
                print(f"IN VAL LOOP 2 | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
            batch_size = len(batch["uid"])
            pbar.update(1)
            if accelerator.process_index not in rank2uid_frame_val:
                rank2uid_frame_val[accelerator.process_index] = []
            rank2uid_frame_val[accelerator.process_index].extend([decode_text(item)+"_"+str(frame.item()) for item,frame in zip(batch["uid"],batch["frame"])])
            accelerator.wait_for_everyone()
        print(f"VAL LOOP 2 END | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        print("*************************************************************************************")
        check_rank2uid_frame(rank2uid_frame_train,epoch, "train")
        check_rank2uid_frame(rank2uid_frame_val,epoch, "val")
        train_dataloader.dataset.dataset.reset_epoch(epoch+1)
        print(f"reset Train at end of epoch {epoch} | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        val_dataloader.dataset.dataset.reset_epoch(epoch+1)
        print(f"reset Val at end of epoch {epoch} | rank {accelerator.process_index} |  {get_detailed_thread()}",flush=True)
        ## 打印内存和线程数：正常情况下，内存和线程数应该保持基本稳定
        print(f"rank {accelerator.process_index} after reset epoch {epoch+1} data memory usage: {get_detailed_memory_usage()}",flush=True)
        print('acum_time ', acum_time, flush=True)
        print("*************************************************************************************")
    ############# check data #############
    if accelerator.process_index == 0:
        results = analyze_data_distribution(epoch=1)

        def print_split_analysis(split_name, split_data):
            print(f"\n{split_name}数据分析:")
            print(f"总uid数量: {split_data['total_uids']}")
            
            # 检查重复数据
            if split_data['overlapping']:
                print(f"\n{split_name}中的重复数据：")
                for ranks, duplicates in split_data['overlapping'].items():
                    print(f"{ranks}: {len(duplicates)} 条重复")
            else:
                print(f"\n{split_name}中没有发现重复数据")
            
            # 统计frame缺失情况
            print(f"\n{split_name} frame缺失统计:")
            missing_ratios = [stats['missing_ratio'] for stats in split_data['frame_stats'].values()]
            if missing_ratios:
                avg_missing_ratio = sum(missing_ratios) / len(missing_ratios)
                print(f"平均缺失比例: {avg_missing_ratio:.2%}")
                
                # 按缺失比例分布统计
                ratio_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
                for start, end in ratio_ranges:
                    count = sum(1 for r in missing_ratios if start <= r < end)
                    print(f"缺失比例 {start:.1%}-{end:.1%}: {count}个uid")
                

        # 分析train数据
        print_split_analysis("Train", results['train'])

        # 分析val数据
        print_split_analysis("Val", results['val'])

        # 检查train和val重合
        print("\nTrain和Val重合分析:")
        print(f"重合uid数量: {results['train_val_overlap']['overlap_count']}")
        if results['train_val_overlap']['overlap_count'] > 0:
            print("重合的uid示例:")
            for uid in list(results['train_val_overlap']['overlapping_uids'])[:5]:  # 只显示前5个
                print(f"- {uid}")

    accelerator.wait_for_everyone()

from x2robot_dataset.common.replay_buffer import ReplayBuffer
import zarr
from x2robot_dataset.common.imagecodecs_numcodecs import JpegXl
from collections import defaultdict
from typing import List, Dict
import numpy as np
import av
import tqdm
import concurrent.futures
import os
import multiprocessing
from multiprocessing import Event
import json

from x2robot_dataset.common.data_preprocessing import (
    MaskMetaData,
    process_videos,
    process_action,
    process_instruction,
    process_tactility,
    check_files,
    trim_stationary_ends,
    LabelData,
    _ACTION_KEY_EE_MAPPING_INV,
    _TAC_FILE_MAPPING,
    _HEAD_ACTION_MAPPING,
    _HEAD_ACTION_MAPPING_INV,
)

import glob


import functools
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, MaskMetaData):
            return {'mask_meta_file_path': obj.file_path}  # 假设 MaskMetaData 有一个 path 属性
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

def capture_inputs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inputs = {
            'args': args,
            'kwargs': kwargs
        }
        
        # 获取 output_dir 参数的值
        output_dir = kwargs.get('output_dir', '')
        
        # 获取 output_dir 的父目录
        parent_dir = os.path.dirname(output_dir)
        
        # 构建 meta.json 文件的完整路径
        meta_path = os.path.join(parent_dir, 'meta.json')
        
        # 创建 parent_dir 目录（如果不存在）
        os.makedirs(parent_dir, exist_ok=True)
        
        # 将输入参数保存到 meta.json 文件
        with open(meta_path, 'w') as f:
            json.dump(inputs, f, indent=2, cls=CustomJSONEncoder)
        
        return func(*args, **kwargs)
    
    return wrapper

class X2ReplayBuffer:
    def __init__(self) -> None:
        out_replay_buffer = ReplayBuffer.create_empty_zarr(
                storage=zarr.MemoryStore())
        video_buffer = defaultdict(lambda:[])
        tactile_buffer = defaultdict(lambda:[])

        self.out_replay_buffer = out_replay_buffer
        self.video_buffer = video_buffer
        self.tactile_buffer = tactile_buffer
        

    def create_zarr(self,
                    video_files:List[List[str]],
                    cam_mapping={'faceImg':'face_view',
                                'leftImg':'left_wrist_view',
                                'rightImg':'right_wrist_view'
                                },
                    image_height=480,
                    image_width=640,
                    compression_level=99,
                    tac_mapping=_TAC_FILE_MAPPING,
                    tac_dim=(3,15),
                    head_action_dim = (2,),
                    filter_angle_outliers=True,
                    detect_motion=True,
                    trim_stationary=True,
                    parse_tactile=False,
                    parse_head_action=False,):
        
        out_res = (image_height, image_width)

        buffer_start = 0

        report_file = os.path.join(os.path.dirname(video_files[0]), 'report.json')
        report_file = report_file if os.path.exists(report_file) else None

        valid_video_files, ee_osc_files = check_files(video_files,
                cam_list=cam_mapping.keys(),
                detect_phase_switch=filter_angle_outliers,
                detect_motion=detect_motion,
                report_file=report_file)
        
        uids = np.array([file.split('/')[-1] for file in valid_video_files])
        self.out_replay_buffer.add_episode(data={'sample_names':uids}, compressors=None)
        print(f'replay_buffer: {self.out_replay_buffer}')

        tactiles = []
        masks = []
        # print(f'valid_video_files: {valid_video_files}')
        for valid_path in valid_video_files:
            uid = valid_path.split('/')[-1]
            filter_angle_outliers = filter_angle_outliers and (uid in ee_osc_files)
            inv_map = _ACTION_KEY_EE_MAPPING_INV | _HEAD_ACTION_MAPPING_INV if parse_head_action else _ACTION_KEY_EE_MAPPING_INV
            action_data = process_action(valid_path,
                                         action_key_mapping = inv_map,
                                         filter_angle_outliers=filter_angle_outliers)
            
            if trim_stationary:
                action_data, (trimmed_start, trimmed_end) = trim_stationary_ends(action_data, threshold=0.05)
            else:
                trimmed_start = 0
                trimmed_end = len(action_data[list(action_data.keys())[0]])

            first_action_key = list(action_data.keys())[0]

            self.out_replay_buffer.add_episode(data=action_data, compressors=None)

            if parse_tactile:
                tactile_data = process_tactility(valid_path)
                tactile_data = {key:value[trimmed_start:trimmed_end] for key,value in tactile_data.items()}
                if tactile_data:
                    tactiles.append(tactile_data)
            

            mp4_files = glob.glob(os.path.join(valid_path, '*.mp4'))
            for file in mp4_files:
                file_name = file.split('/')[-1].split('.')[0]
                if file_name not in cam_mapping:
                    continue
                cam_name = cam_mapping[file_name]
                self.video_buffer[file].append({
                    'camera_name': cam_name,
                    'frame_start': trimmed_start,
                    'frame_end': trimmed_end,
                    'buffer_start': buffer_start
                })
            buffer_start += trimmed_end - trimmed_start

        img_compressor = JpegXl(level=compression_level, numthreads=1)
        for name in cam_mapping.values():
            _ = self.out_replay_buffer.data.require_dataset(
                name=name,
                shape=(self.out_replay_buffer[first_action_key].shape[0],) + out_res + (3,),
                chunks=(1,) + out_res + (3,),
                compressor=img_compressor,
                dtype=np.uint8
            )
        
        if parse_tactile:
            assert len(tactiles) == len(valid_video_files), 'Tactile data is missing for some videos'
            
            tactiles = {k: [d[k] for d in tactiles] for k in tactiles[0].keys()}
            for name in tac_mapping.values():
                _ = self.out_replay_buffer.data.require_dataset(
                        name=name,
                        shape=(self.out_replay_buffer[first_action_key].shape[0],) + tac_dim,
                        chunks=(1,) + tac_dim,
                        compressor=None,
                        dtype=np.int16
                )
                print(name, len(tactiles[name]), tactiles[name][0].shape)
                self.out_replay_buffer.data[name] = np.concatenate(tactiles[name], axis=0)

        base_dir = os.path.dirname(video_files[0])
        print(f'{base_dir} buffer:', self.out_replay_buffer, flush=True)
    
    def write_video_to_buffer(self, num_workers = 12):
        def video_to_zarr(mp4_path, task):
            name = task['camera_name']
            img_array = self.out_replay_buffer.data[name]

            with av.open(mp4_path) as container:
                in_stream = container.streams.video[0]
                in_stream.thread_count = 10
                buffer_idx = task['buffer_start'] 

                for idx, frame in enumerate(container.decode(in_stream)):
                    if idx < task['frame_start'] or idx >= task['frame_end']:
                        continue

                    img = frame.to_ndarray(format='rgb24')
                    img_array[buffer_idx] = img
                    buffer_idx += 1

        with tqdm.tqdm(total=len(self.video_buffer.keys())) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = set()
                for mp4_path, tasks in self.video_buffer.items():
                    for task in tasks:
                        if len(futures) >= num_workers:
                            completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                            pbar.update(len(completed))
                        
                        future = executor.submit(video_to_zarr, mp4_path, task)
                        futures.add(future)

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))
    
    def save_to_disk(self, output_dir: str):
        print(f"Saving ReplayBuffer to {output_dir}", flush=True)
        with zarr.ZipStore(output_dir, mode='w') as zip_store:
            self.out_replay_buffer.save_to_store(store=zip_store)


    @staticmethod
    @capture_inputs
    def make_zarr(video_files:List[List[str]],
                    cam_mapping={'faceImg':'face_view',
                                'leftImg':'left_wrist_view',
                                'rightImg':'right_wrist_view'
                                },
                    image_height=480,
                    image_width=640,
                    compression_level=99,
                    output_dir:str='./tmp.zarr',
                    parse_tactile=False,
                    parse_head_action=False,
                    num_workers = 24,
                    filter_angle_outliers=True,
                    detect_motion=True,
                    trim_stationary=True,):
        x2_replay_buffer = X2ReplayBuffer()
        x2_replay_buffer.create_zarr(video_files=video_files,
                                    cam_mapping=cam_mapping,
                                    image_height=image_height,
                                    image_width=image_width,
                                    compression_level=compression_level,
                                    filter_angle_outliers=filter_angle_outliers,
                                    detect_motion=detect_motion,
                                    trim_stationary=trim_stationary,
                                    parse_tactile=parse_tactile,
                                    parse_head_action=parse_head_action,)
        x2_replay_buffer.write_video_to_buffer(num_workers=num_workers)
        x2_replay_buffer.save_to_disk(output_dir=output_dir)

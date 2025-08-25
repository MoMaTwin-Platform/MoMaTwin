import torchvision
import torch
from einops import rearrange
import gc
import numpy as np
import av
# def decode_video_torchvision(file_name, keyframes_only=True, backend = 'pyav'):
#     torchvision.set_video_backend(backend)

#     reader = torchvision.io.VideoReader(file_name, "video")
#     reader.seek(0, keyframes_only=keyframes_only)

#     loaded_frames = []
#     for frame in reader:
#         current_ts = frame["pts"]
#         loaded_frames.append(frame["data"])

#     loaded_frames = torch.stack(loaded_frames)
#     if loaded_frames.shape[-1] != 3:
#         loaded_frames = rearrange(loaded_frames, 'b c h w -> b h w c')

#     loaded_frames = loaded_frames.numpy()

#     reader.container.close()
#     for stream in reader.container.streams:
#         stream.codec_context.close()
#     reader = None

#     return loaded_frames

def decode_video_torchvision(file_name, frame_indices=None, keyframes_only=True, backend='pyav'):
    """加载视频指定帧
    Args:
        file_name: 视频文件路径
        frame_indices: 需要加载的帧索引列表，如果为None则加载所有帧
        keyframes_only: 是否只读取关键帧
        backend: 视频读取后端
    Returns:
        loaded_frames: numpy数组，shape为(n_frames, height, width, 3)
    """
    # 首先使用 PyAV 获取视频信息
    with av.open(file_name) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or (stream.time_base.denominator / stream.time_base.numerator))
        total_frames = stream.frames

    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(file_name, "video")

    try:
        if frame_indices is None:
            # 如果不指定frame_indices，加载所有帧
            reader.seek(0, keyframes_only=keyframes_only)
            loaded_frames = []
            for frame in reader:
                loaded_frames.append(frame["data"])
        else:
            # 验证frame_indices的有效性
            frame_indices = sorted([i for i in frame_indices if 0 <= i < total_frames])
            if not frame_indices:
                raise ValueError("No valid frame indices provided")

            loaded_frames = []
            for target_idx in frame_indices:
                # 计算需要seek的时间戳（以秒为单位）
                target_ts = float(target_idx) / fps
                reader.seek(int(target_ts * 1000))  # VideoReader需要毫秒为单位
                
                frame = next(reader)
                loaded_frames.append(frame["data"])

        # 转换为numpy数组
        loaded_frames = torch.stack(loaded_frames)
        if loaded_frames.shape[-1] != 3:
            loaded_frames = rearrange(loaded_frames, 'b c h w -> b h w c')
        loaded_frames = loaded_frames.numpy()

    finally:
        # 确保资源被正确释放
        reader.container.close()
        for stream in reader.container.streams:
            stream.codec_context.close()
        reader = None

    return loaded_frames


# def decode_video_torchvision_frame(file_name, keyframes_only=True, backend = 'pyav', frame_idx=[0]):
#     torchvision.set_video_backend(backend)
#     print("reading video", file_name, flush=True)

#     reader = torchvision.io.VideoReader(file_name, "video")
#     time_base = reader.container.streams[0].time_base
#     timestamps = [frame_idx[i]/time_base for i in range(len(frame_idx))]
#     start_frame_ts = timestamps[0]
#     end_frame_ts = timestamps[-1]
#     print(f'start_frame_ts:{start_frame_ts} end_frame_ts:{end_frame_ts}', flush=True)

#     reader.seek(start_frame_ts, keyframes_only=keyframes_only) # find start key frame

#     loaded_frames = []
#     loaded_frame_ts = []
#     for frame in reader:
#         current_ts = frame["pts"]
#         loaded_frames.append(frame["data"])
#         loaded_frame_ts.append(current_ts)
#         if current_ts >= end_frame_ts:
#             break

#     reader.container.close()
#     loaded_frames = torch.stack(loaded_frames)
#     if loaded_frames.shape[-1] != 3:
#         loaded_frames = rearrange(loaded_frames, 'b c h w -> b h w c')

#     query_ts = torch.tensor(timestamps)
#     loaded_ts = torch.tensor(loaded_frame_ts)

#     # compute distances between each query timestamp and timestamps of all loaded frames
#     dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
#     min_, argmin_ = dist.min(1)
#     # get closest frames to the query timestamps
#     closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
#     closest_ts = loaded_ts[argmin_]


#     for stream in reader.container.streams:
#         print("close stream", flush=True)
#         stream.codec_context.close()
#     reader = None

#     closest_frames = closest_frames.numpy()

#     print(f'loaded_frames:{loaded_frames.shape} closest_frames:{closest_frames.shape}', flush=True)

#     return closest_frames
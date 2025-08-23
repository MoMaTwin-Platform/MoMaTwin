import json
import cv2
import numpy as np
from typing import Union
import torch


class InstructPutWaypoint:
    def __init__(self, file: str, custom_defined_invalid_samples: str):
        with open(custom_defined_invalid_samples, 'r') as f:
            self.custom_defined_invalid_samples = json.load(f)
        with open(file, 'r') as f:
            self.visual_instruct_dict = json.load(f)
        # 如果是waypoint模式，那么字典self.visual_instruct_dict的key是sample, value是对应的waypoint视频的第一帧，需要将每个视频的第一帧提取出来放到内存中
        new_visual_instruct_dict = {}
        for sample, waypoint_video_path in self.visual_instruct_dict.items():
            waypoint_video = cv2.VideoCapture(waypoint_video_path)
            waypoint_frame = waypoint_video.read()[1]
            waypoint_frame = cv2.cvtColor(waypoint_frame, cv2.COLOR_BGR2RGB)
            new_visual_instruct_dict[sample] = waypoint_frame
        self.visual_instruct_dict = new_visual_instruct_dict

    def decode_text(self, encoded_array:Union[torch.Tensor, np.ndarray]) -> str:
        '''
        Decode text from encoded array
        '''
        if isinstance(encoded_array, str):
            return encoded_array
        if len(encoded_array) == 0:
            return ''
        
        if isinstance(encoded_array, torch.Tensor):
            encoded_array = encoded_array.cpu().numpy()
        
        # 过滤掉 0 并转换为字符
        filtered_text = encoded_array[encoded_array != 0]

        return ''.join(map(chr, filtered_text))

    def fuse_waypoint_img(self, batch):
        face_view_imgs = batch['obs']['face_view']  # [B, T, H, W, C]
        samples_with_prefix = [self.decode_text(item) for item in batch["uid"]]
        samples = [item.replace(item.split('_')[0] + '_', '') for item in samples_with_prefix]
        batch_size = face_view_imgs.shape[0]
        wrong_cnt = 0
        for i in range(batch_size):
            face_view_img = face_view_imgs[i][0]
            waypoint_img = None
            if samples[i] not in self.visual_instruct_dict:
                print(f"Sample {samples[i]} not found in visual_instruct_dict")
                wrong_cnt += 1
                continue
            way_point_img = self.visual_instruct_dict[samples[i]]
            alpha = 0.6
            fused_img = cv2.addWeighted(face_view_img.cpu().numpy(), alpha, way_point_img, 1-alpha, 0)
            fused_img = torch.from_numpy(fused_img).to(face_view_img.device)
            batch['obs']['face_view'][i][0] = fused_img

        return batch


    def modify_obs(self, batch):
        batch = self.fuse_waypoint_img(batch)
        return batch
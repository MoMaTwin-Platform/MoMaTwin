import json
import cv2
import numpy as np
from typing import Union
import torch


class InstructPutBox:
    def __init__(self, file: str, custom_defined_invalid_samples: str):
        with open(custom_defined_invalid_samples, 'r') as f:
            self.custom_defined_invalid_samples = json.load(f)
        with open(file, 'r') as f:
            self.visual_instruct_dict = json.load(f)

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


    def draw_boxes_on_img(self, img, boxes, colors):
        """
        在图像上绘制半透明颜色的边界框（直接操作 uint8 张量）
        
        参数:
            img (torch.Tensor): 形状为 (h, w, c) 的 uint8 图像张量
            boxes (list): 边界框列表，每个元素是 [x1, y1, x2, y2]
            colors (list): 颜色列表，每个元素是 [b, g, r]
        
        返回:
            torch.Tensor: 绘制了边界框的 uint8 图像
        """
        # 确保输入是 uint8 类型
        assert img.dtype == torch.uint8, "输入图像必须是 uint8 类型"
        
        # 创建图像的副本以避免修改原始图像
        img_with_boxes = img.clone()
        
        # 设置透明度
        alpha = 0.5  # 半透明
        
        for box, color in zip(boxes, colors):
            x1, y1, x2, y2 = box
            # 将颜色转换为 uint8 张量
            color_tensor = torch.tensor(color, dtype=torch.uint8, device=img.device)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(int(x1), img.shape[1] - 1))
            y1 = max(0, min(int(y1), img.shape[0] - 1))
            x2 = max(0, min(int(x2), img.shape[1] - 1))
            y2 = max(0, min(int(y2), img.shape[0] - 1))
            
            # 如果框无效则跳过
            if x1 >= x2 or y1 >= y2:
                continue
            
            # 应用半透明颜色（直接在 uint8 上操作）
            img_with_boxes[y1:y2, x1:x2, :] = (
                (alpha * color_tensor.view(1, 1, 3)).to(torch.uint8) + 
                ((1 - alpha) * img_with_boxes[y1:y2, x1:x2, :]).to(torch.uint8)
            )
        
        return img_with_boxes


    def add_box(self, batch):
        face_view_imgs = batch['obs']['face_view']  # [B, T, H, W, C]
        samples_with_prefix = [self.decode_text(item) for item in batch["uid"]]
        samples = [item.replace(item.split('_')[0] + '_', '') for item in samples_with_prefix]
        frames = batch['frame']
        batch_size = face_view_imgs.shape[0]
        wrong_cnt = 0
        for i in range(batch_size):
            frame = int(frames[i])
            if samples[i] not in self.visual_instruct_dict:
                # print(f"Sample {samples[i]} not found in visual_instruct_dict")
                wrong_cnt += 1
                continue
            sample = samples[i]
            img = face_view_imgs[i][0]
            if str(frame) not in self.visual_instruct_dict[sample]:
                # print(f"Frame {frame} not found in visual_instruct_dict for sample {samples[i]}")
                wrong_cnt += 1
                continue
            box_info = self.visual_instruct_dict[sample][str(frame)]
            colors = [[0, 0, 255], [255, 0, 0]]
            img_with_box = self.draw_boxes_on_img(img, box_info, colors)
            batch['obs']['face_view'][i][0] = img_with_box

        return batch


    def modify_obs(self, batch):
        batch = self.add_box(batch)
        return batch
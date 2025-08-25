import json
import cv2
import numpy as np
from typing import Union
import torch
from tqdm import tqdm

class EntangleLineFuseView:
    def __init__(self, file: str, custom_defined_invalid_samples: str):
        with open(custom_defined_invalid_samples, 'r') as f:
            self.custom_defined_invalid_samples = json.load(f)
        with open(file, 'r') as f:
            self.visual_instruct_dict = json.load(f)
        new_visual_instruct_dict = {}
        print('读取绕线数据box信息')
        for sample, dic_ls in tqdm(self.visual_instruct_dict.items()):
            video_path = dic_ls[0]['video_path']
            video = cv2.VideoCapture(video_path)
            first_frame = video.read()[1]
            new_visual_instruct_dict[sample] = {}
            for dic in dic_ls:
                box = dic['box']
                box = [int(x) for x in box]
                first_frame_cropped = first_frame[box[1]:box[3], box[0]:box[2]]
                start_index = dic['start_index']
                end_index = dic['end_index']
                new_visual_instruct_dict[sample][f'{start_index}_{end_index}'] = {
                    'cropped_box_img': first_frame_cropped,
                    'box': box
                }
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

    def fuse_view_entangle_line(self, batch):
        face_view_imgs = batch['obs']['face_view']  # [B, T, H, W, C]
        samples_with_prefix = [self.decode_text(item) for item in batch["uid"]]
        samples = [item.replace(item.split('_')[0] + '_', '') for item in samples_with_prefix]
        frames = batch['frame']
        batch_size = face_view_imgs.shape[0]
        wrong_cnt = 0
        for i in range(batch_size):
            face_view_img = face_view_imgs[i][0]
            if samples[i] not in self.visual_instruct_dict:
                # print(f"Sample {samples[i]} not found in visual_instruct_dict")
                wrong_cnt += 1
                continue
            box_info = self.visual_instruct_dict[samples[i]]
            cur_frame_index = int(frames[i])
            target_box = None
            face_view_cur_frame = face_view_img.cpu().numpy()

            flag = True
            for key, sub_dic in box_info.items():
                state_index, end_index = int(key.split('_')[0]), int(key.split('_')[1])
                bgr_box_npy = sub_dic['cropped_box_img']
                box = sub_dic['box']
                if box[0] >= box[2] or box[1] >= box[3]:
                    flag = False
                    break
                rgb_box_img = cv2.cvtColor(bgr_box_npy, cv2.COLOR_BGR2RGB)
                x1, y1, x2, y2 = map(int, box)
                tmp_area = face_view_cur_frame.copy()[y1:y2, x1:x2]
                alpha = 0.5
                blended = cv2.addWeighted(tmp_area, alpha, rgb_box_img, 1-alpha, 0)
                face_view_cur_frame[y1:y2, x1:x2] = blended
                if cur_frame_index >= state_index and cur_frame_index < end_index:
                    target_box = box

            if not flag:
                # print('box size error: box[0] >= box[2] or box[1] >= box[3] : ', samples[i])
                continue

            if target_box is None:
                # print(f"Frame {cur_frame_index} not found in visual_instruct_dict for sample {samples[i]}")
                wrong_cnt += 1
                continue

            x1, y1, x2, y2 = map(int, target_box)
            face_view_cur_frame = np.ascontiguousarray(face_view_cur_frame)
            face_view_cur_frame = cv2.rectangle(face_view_cur_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # cv2.imwrite(f'/x2robot/caichuang/extra_codes/dp_roy/diffusion_policy/test_img/a/{i}.jpg', face_view_cur_frame)
            
            batch['obs']['face_view'][i][0] = torch.from_numpy(face_view_cur_frame).to(face_view_img.device)

        # print(f"wrong_cnt:{wrong_cnt} | batch size:{batch_size}")
        return batch


    def modify_obs(self, batch):
        batch = self.fuse_view_entangle_line(batch)
        return batch
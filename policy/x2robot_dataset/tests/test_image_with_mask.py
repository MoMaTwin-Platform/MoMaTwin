from torch.utils.data import DataLoader
from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
import tqdm

from x2robot_dataset.common.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _CAM_FILE_MAPPING,
    _HEAD_ACTION_MAPPING,
)
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from accelerate import Accelerator
def put_text_multiline(img, text, org, font_face, font_scale, color, thickness, line_spacing=1.5):
    lines = text.split('\n')
    y = org[1]
    for line in lines:
        textsize = cv2.getTextSize(line, font_face, font_scale, thickness)[0]
        
        # 如果文本宽度超过图像宽度，进行自动换行
        max_width = img.shape[1] - org[0] - 10  # 留出10像素的边距
        if textsize[0] > max_width:
            words = line.split()
            new_line = words[0]
            for word in words[1:]:
                test_line = new_line + ' ' + word
                test_size = cv2.getTextSize(test_line, font_face, font_scale, thickness)[0]
                if test_size[0] <= max_width:
                    new_line = test_line
                else:
                    cv2.putText(img, new_line, (org[0], y), font_face, font_scale, color, thickness, cv2.LINE_AA)
                    y += int(textsize[1] * line_spacing)
                    new_line = word
            cv2.putText(img, new_line, (org[0], y), font_face, font_scale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(img, line, (org[0], y), font_face, font_scale, color, thickness, cv2.LINE_AA)
        
        y += int(textsize[1] * line_spacing)
    
    return img

def main():
    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size=accelerator.num_processes
    print(f'{rank}/{world_size}')
    # dataset_paths = ['/x2robot/zhengwei/10000/20240827-pick_up-item']
    dataset_paths = ['/x2robot/zhengwei/10000/20240827-pick_up-item']
    # dataset_paths = ['/x2robot/zhengwei/10000/20240813-pick_up-item']
    default_data_configs = {'default_instruction': 'pick up item onto container.',
                            'action_horizon': 100,
                            'action_history_length': 0,
                            'image_horizon': 1,
                            'image_history_length': 0,
                            'right_padding': True,
                            'left_padding': False,
                            'train_val_split': 0.9,
                            'split_seed': 42,
                            'obs_keys': list(_CAM_FILE_MAPPING.keys()),
                            'action_keys': list(_ACTION_KEY_EE_MAPPING.keys()),
                            }

    # mask_meta_file_path = '/x2robot/caichuang/pickup_data_all_0905_compress.json'
    mask_meta_file_path = '/x2robot/caichuang/pickup_json_files/pick_up_data_0929_with_quality_compressed_mask.json'
    # mask_type = 'mask_on_image'
    # mask_type = 'box_on_image'
    mask_type = 'mask_only'
    mask_keys = ['face_mask', 'left_mask', 'right_mask']
    train_dataset, val_dataset, train_sampler, val_sampler = make_chunk_dataset(
                       dataset_paths,
                       rank=rank,
                       dp_world_size=world_size,
                       cache_dir='/x2robot/Data/.cache_mask', #缓存数据集的地址，可以被其它进程共享
                       read_labeled_data=False, #是否读取人工标注的数据
                       trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                       read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                       memory_size=8, #内存使用大小, GB
                       dataset_buffer_size=20, #数据集缓冲区大小
                       parse_tactile=False, #是否解析触觉数据
                       parse_head_action=False, #是否解析头部数据
                       return_full_traj=False, #是否返回完整轨迹, default False
                       flush_cache=False, #是否清空缓存
                       data_configs=[default_data_configs] * len(dataset_paths), #数据配置
                       num_workers=32, #数据处理的进程数
                       filter_angle_outliers=True, #是否过滤角度异常值
                       detect_motion=True, #是否去掉静止不动的样本
                       trim_stationary=False, #是否去掉首尾不动的部分
                       mask_meta_file_path=mask_meta_file_path,
                       mask_type=mask_type,
                       mask_in_buffer=False,
                       mask_keys=mask_keys) # 是否把mask提前存到replay buffer里

    collate_fn = collate_wrapper(obs_keys=default_data_configs['obs_keys'],
                                 collate_type = 'chunking', #chuning, full
                                 low_dim_obs_horizon=1,
                                 img_obs_horizon=1,
                                 horizon=20,
                                 action_dim=14, # add head_action [2]
                                 is_bi_mode=True,
                                 mask_type=mask_type,
                                 mask_keys=mask_keys)

    data_loader = DataLoader(train_dataset,
                             batch_size=64,
                             sampler=train_sampler,
                             num_workers=4,
                             collate_fn=collate_fn)
    data_loader = accelerator.prepare(data_loader)
    print(f'len:{len(data_loader)}', flush=True)
    cc = 0
    import numpy as np
    tepoch = tqdm.tqdm(total=len(data_loader), desc=f"rank {rank} - ", leave=False, disable=not accelerator.is_local_main_process)
    for batch_idx, batch in enumerate(data_loader): 
        # if not accelerator.is_main_process:
        #     continue
        # tepoch.update(1)
        print(f'batch: {batch.keys()}', flush=True)
        batched_uid = batch['uid']
        batched_text = batch['instruction']
        batched_face_img = batch['obs']['face_view'].cpu().numpy()
        batched_left_img = batch['obs']['left_wrist_view'].cpu().numpy()
        batched_right_img = batch['obs']['right_wrist_view'].cpu().numpy()
        batched_frame = np.array(batch['frame'])
        # for mask_key in mask_keys:
        batched_mask_left = batch['left_mask'].cpu().numpy()
        batched_mask_right = batch['right_mask'].cpu().numpy()
        batched_mask_face = batch['face_mask'].cpu().numpy()
        for uid, face_img,left_img,right_img,face_mask,left_mask,right_mask,text,frame_id in zip(batched_uid,batched_face_img,batched_left_img,batched_right_img,batched_mask_face, batched_mask_left,batched_mask_right,batched_text, batched_frame):
            print(f'{uid}\n {text[0]}\n {face_img.shape}\n frameid:{frame_id}')
            face_mask_image = cv2.cvtColor(face_mask[0], cv2.COLOR_RGB2BGR)
            left_mask_image = cv2.cvtColor(left_mask[0], cv2.COLOR_RGB2BGR)
            right_mask_image = cv2.cvtColor(right_mask[0], cv2.COLOR_RGB2BGR)
            face_img = cv2.cvtColor(face_img[0], cv2.COLOR_RGB2BGR)
            left_img = cv2.cvtColor(left_img[0], cv2.COLOR_RGB2BGR)
            right_img = cv2.cvtColor(right_img[0], cv2.COLOR_RGB2BGR)
            # 添加文字到图片
            # 定义文字的位置（左下角坐标）
            position = (50, 50)
            # 定义字体
            font = cv2.FONT_HERSHEY_SIMPLEX
            face_image_with_text = put_text_multiline(face_img, f'{text[0]}_{frame_id}', position, font, 0.5, (0,255,0), 1)
            face_row = cv2.hconcat([face_image_with_text, face_mask_image])
            left_row = cv2.hconcat([left_img, left_mask_image])
            right_row = cv2.hconcat([right_img, right_mask_image])
            cv2.imwrite(f'{uid}.jpg', cv2.vconcat([face_row, left_row, right_row]))
        break

    accelerator.wait_for_everyone()

if __name__ == '__main__':
    main()
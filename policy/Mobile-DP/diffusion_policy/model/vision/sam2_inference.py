# %%
import numpy as np
import torchvision.transforms as T
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor
import torch
import time
import os
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def showcv2image(img_npy):
    img_rgb = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
    img_rgb = Image.fromarray(img_rgb)
    # img_rgb.show()
    return img_rgb

def drawbox(image, boxes) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    # Iterate over detections and add bounding boxes and masks
    image_cv2 = image.copy()
    for box in boxes:
        color = (255,0,0)
        # Draw bounding box
        cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), color, 2)
        # cv1.putText(image_cv2, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
    return image_cv2

def drawmask(image, masks, alpha=0.3) -> np.ndarray:
    image_cv2 = image.copy()
    # Iterate over detections and add bounding boxes and masks

    for mask in masks:
        # cv2 is BGR, static is blue
        # color = (0,0,255)
        color = (255,0,0)
        # Draw masks
        overlay = np.zeros(image_cv2.shape, dtype=np.uint8)
        overlay[mask == 1] = color  # 红色覆盖区域
        
        # 将颜色叠加层和原始图像混合
        image_cv2 = cv2.addWeighted(overlay, alpha, image_cv2, 1 - alpha, 0)
    return image_cv2

def vis_mask_on_video(mask, video_path, save_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建 VideoWriter 以保存新视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    # 逐帧处理视频
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 从 mask 中获取当前帧的 mask
        current_mask = mask[i] if i < len(mask) else None
        
        # 如果有 mask 数据，则在当前帧上绘制红色覆盖区域
        if current_mask is not None:
            overlay = np.zeros_like(frame)
            overlay[current_mask] = (0, 0, 255)  # 红色覆盖区域
            
            # 将 overlay 叠加到 frame 上
            alpha = 0.5  # 叠加的透明度
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 将处理后的帧写入输出视频文件
        out.write(frame)
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"Video with mask visualization saved to {save_path}")


class Agent:
    def __init__(self, 
        sam_model_path = '/x2robot/caichuang/weights/MobileSAM/mobile_sam.pt',
    ):
        self.mobile_sam = sam_model_registry['vit_t'](checkpoint=sam_model_path)
        model_id = "/x2robot/ganruyi/models/grounding-dino-base"
        self.device = 'cuda'
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.mobile_sam.to(self.device)
        self.mobile_sam.eval()
        self.predictor = SamPredictor(self.mobile_sam)
        from sam2.build_sam import build_sam2_camera_predictor
        checkpoint = "/x2robot/ganruyi/models/sam2_hiera/sam2_hiera_large.pt"
        # model_cfg = "/x2robot/ganruyi/models/sam2_hiera/sam2_configs/sam2_hiera_l.yaml"
        model_cfg = "sam2_hiera_l.yaml"
        self.sam2_predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

        self.start = False
        self.initial_masks = []
        self.initial_points = []

    def get_mask_from_box(self, cur_frame, box_npy):
        self.predictor.set_image(cur_frame, image_format="BGR")
        masks, _, _ = self.predictor.predict(box=box_npy)
        return masks[0] # (h, w)

    def detect(self, img_npy, text, box_threshold = 0.3, text_threshold = 0.25):
        # img_npy is RGB mode, the same as training, H,W,C
        # target is label list
        if len(text) > 0:
            text += '.'
        print(f'text:{text}')
        inputs = self.processor(images=img_npy, text=text, return_tensors='pt', padding='longest').to(self.device)
        B = img_npy.shape[0]
        results = None
        with torch.no_grad():
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[img_npy.shape[:-1]]
            )
            # print(results)
        return results        

    def get_box(self, cur_frame, target_ls):
        # cur_frame: numpy array (H, W, 2), RGB mode
        # target_ls: list of str [red cup, green plate]
        res_boxes = []
        for target in target_ls:
            result = self.detect(cur_frame, target)
            scores = result[0]['scores'].cpu().numpy()
            if len(scores) > 0:
                max_score_idx = scores.argmax()
                box = result[0]['boxes'].cpu().numpy().astype(int).tolist()[max_score_idx]
                res_boxes.append(box)
            else:
                res_boxes.append([0,0,0,0])
        return res_boxes

    def get_mask(self, cur_frame, target_ls):
        res_masks = []
        boxes = self.get_box(cur_frame, target_ls)
        assert len(boxes) == 2, f'len(boxes): {len(boxes)}'
        # if len(boxes) == 0:
            # res_masks.append(np.zeros(cur_frame.shape[:-1], dtype=np.uint8))
            # return res_masks 
        for box in boxes:
            mask = self.get_mask_from_box(cur_frame, np.array(box))
            res_masks.append(mask)
        return res_masks

    def sample_points_from_mask(self, mask, num):
        true_indices = np.argwhere(mask)
        if len(true_indices) < num:
            return true_indices.tolist()
        sampled_indices = true_indices[np.random.choice(len(true_indices), num, replace=False)]
        return sampled_indices.tolist()
    
    def get_mask_sam2_online(self, cur_frame, target_ls):
        res_masks = []
        if not self.start:
            # 通过grounding dino + mobilesam获取sam2的起始跟踪点（存在self.initial_points中）
            self.start = True
            boxes = self.get_box(cur_frame, target_ls)
            # assert len(boxes) == 2, f'len(boxes): {len(boxes)}'
            for box in boxes:
                mask = self.get_mask_from_box(cur_frame, np.array(box))
                self.initial_masks.append(mask)
                points = self.sample_points_from_mask(mask, 10)
                points = [[x[1], x[0]] for x in points]
                self.initial_points.append(points)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

                # 初始化sam2_predictor
                self.sam2_predictor.load_first_frame(cur_frame)

                # 初始帧sam2 mask生成
                for i in range(len(self.initial_points)):
                    points = self.initial_points[i]
                    # 此时的box是一个二维列表，代表一个目标上的若干个点
                    # 例：[[1,2], [3,4], [5,6]]
                    frame_idx = 0    # 这个是固定的，代表我们选取的点是从初始帧开始跟踪
                    obj_id = i       # 一个目标对应一个独特的ID
                    labels = np.array([1]*len(points), np.int32)   # labels个数要和点的个数相同
                    _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_points(frame_idx, obj_id, points, labels)
                    res_masks.append(out_mask_logits[0][0].cpu().numpy() > 0)
                
                return res_masks
        
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # 后续帧不再需要grounding dino与mobile sam，直接用sam2_predictor
                out_obj_ids, out_mask_logits = self.sam2_predictor.track(cur_frame)
                for i in range(len(out_mask_logits)):
                    res_masks.append(out_mask_logits[i][0].cpu().numpy() > 0)

                return res_masks

if __name__ == '__main__':
    # image_path = '/home/ganruyi/code/10001##20240614-pick_up-drink##20240614-pick_up-drink@MASTER_SLAVE_MODE@2024_06_14_21_46_34##leftImg##131.jpg'
    # img_cv2 = cv2.imread(image_path)
    # img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    agent = Agent()
    # target_ls = ['water bottle', 'red plate']
    # boxes = agent.get_box(cur_frame=img_rgb, target_ls=target_ls)
    # Image.fromarray(drawbox(img_rgb, boxes=boxes))

    # mask = agent.get_mask(cur_frame=img_rgb, target_ls=target_ls)
    # Image.fromarray(drawmask(img_rgb, masks=mask))

    # 视频验证
    video_path = '/x2robot/zhengwei/10000/20240823-pick_up-item/20240823-pick_up-item@MASTER_SLAVE_MODE@2024_08_23_17_46_58/faceImg.mp4'
    cap = cv2.VideoCapture(video_path)

    res_masks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # res_mask = agent.get_mask_sam2_online(frame, ['toothpaste'])[0]
        res_mask = agent.get_mask_sam2_online(frame, ['purple onion'])[0]
        res_masks.append(res_mask)


    vis_mask_on_video(res_masks, video_path, 'face.mp4')
# %%

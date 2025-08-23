import copy
import numpy as np
import torchvision.transforms as T
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor
import torch
import re
import os
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_camera_predictor
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TARGET_EXTRACT_ERROR = "TARGET_EXTRACT_ERROR"
DES_AUG_ERROR = "DES_AUG_ERROR"
NO_DETECTION_ERROR = "NO_DETECTION_ERROR"

def extract_box_coordinates(text):
    # 使用正则表达式查找符合 (x1,y1),(x2,y2) 格式的 box 坐标
    match = re.search(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>", text)
    
    if match:
        # 提取匹配的坐标，并将其转换为整数
        x1, y1, x2, y2 = map(int, match.groups())
        return [x1, y1, x2, y2]
    else:
        # 如果没有找到符合格式的坐标，返回 None
        return None

def save_imgs_as_video(img_ls, video_path, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (img_ls[0].shape[1], img_ls[0].shape[0]))
    for img in img_ls:
        out.write(img)
    out.release()

def viz_boxes(img, boxes, colors):
    color_dict = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
    }
    img_with_boxes = img.copy()
    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = box
        bgr_color = color_dict.get(color.lower(), (0, 0, 255))
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), bgr_color, 2)
    return img_with_boxes

def compute_bounding_boxes(masks):
    num = len(masks)
    bboxes = np.zeros((num, 4), dtype=int)
    for i in range(num):
        indices = np.argwhere(masks[i])
        if indices.size == 0:
            bboxes[i] = [0, 0, 0, 0]
        else:
            y_min, x_min = np.min(indices, axis=0)
            y_max, x_max = np.max(indices, axis=0)
            bboxes[i] = [x_min, y_min, x_max, y_max]
    return bboxes
def video_2_npy(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    video_array = np.array(frames)
    return video_array

def visualize_masks_on_image(mask_ls, img_npy):
    combined_mask = np.zeros_like(img_npy)
    for mask in mask_ls:
        if mask.shape != img_npy.shape[:2]:
            raise ValueError("mask 的形状必须与 img_npy 匹配")
        red_mask = np.zeros_like(img_npy)
        red_mask[mask] = [0, 0, 255]  # BGR格式，红色
        combined_mask = cv2.add(combined_mask, red_mask)
    vis_img = cv2.addWeighted(img_npy, 1.0, combined_mask, 0.8, 0)
    return vis_img

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
        image_cv2 = cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), color, 2)
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

class UniAgent:
    def __init__(self, 
        sam_model_path = '/x2robot/caichuang/weights/MobileSAM/mobile_sam.pt',
    ):
        self.sam_model_path = sam_model_path
        self.mobile_sam = sam_model_registry['vit_t'](checkpoint=self.sam_model_path)
        model_id = "/x2robot/ganruyi/models/grounding-dino-base"
        self.device = 'cuda:1'
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.mobile_sam.to(self.device)
        self.mobile_sam.eval()
        self.predictor = SamPredictor(self.mobile_sam)
        self.checkpoint = "/x2robot/ganruyi/models/sam2_hiera/sam2_hiera_large.pt"
        self.model_cfg = "sam2_hiera_l.yaml"
        # self.sam2_predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint, self.device)
        self.sam2_predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint, self.device)
        # self.cpu_sam2_predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint, 'cpu')

        self.start = False
        self.initial_masks = []
        self.initial_points = []

        self.qwenvl_path = '/x2robot/ganruyi/models/Qwen-VL-Chat'
        # qwenvl_path = '/x2robot/caichuang/weights/Qwen-VL-Chat'
        self.vlm_tokenizer = AutoTokenizer.from_pretrained(self.qwenvl_path, trust_remote_code=True)
        self.vlm = AutoModelForCausalLM.from_pretrained(self.qwenvl_path, device_map=self.device, trust_remote_code=True, bf16=True).eval()
        # self.vlm.cpu()
        self.vlm.generation_config = GenerationConfig.from_pretrained(self.qwenvl_path, trust_remote_code=True)

        self.LLM = OpenAI(api_key="sk-a822e7ebaff84b1193c0fad24a7ec38a", base_url="https://api.deepseek.com")
        self.VLM = None

    def reset(self):
        self.start = False
        self.initial_masks = []
        self.initial_points = []
        # self.sam2_predictor = copy.deepcopy(self.cpu_sam2_predictor).to(self.device)
        # self.sam2_predictor.reset_state()
        self.mobile_sam = sam_model_registry['vit_t'](checkpoint=self.sam_model_path)
        self.mobile_sam.to(self.device)
        self.mobile_sam.eval()
        self.predictor = SamPredictor(self.mobile_sam)
        self.sam2_predictor = build_sam2_camera_predictor(self.model_cfg, self.checkpoint, device=self.device)
        
        # self.free_gpu_memory()
        # self.vlm = AutoModelForCausalLM.from_pretrained(self.qwenvl_path, device_map=self.device, trust_remote_code=True, bf16=True).eval()


    def free_gpu_memory(self):
        if self.vlm is not None:
            # 1. 将模型移动到 CPU
            # self.vlm.cpu()
            
            # 2. 删除模型对象
            del self.vlm
            self.vlm = None
            
            # 3. 清空 CUDA 缓存
            torch.cuda.empty_cache()
        
        print("GPU 内存已释放")
        
    def get_box_using_vlm(self, cur_frame_path, target):
        query = self.vlm_tokenizer.from_list_format([
            {'image': cur_frame_path},
            {'text': '框出图中' + target + '的位置'},
        ])
        # self.vlm.cuda()
        response, history = self.vlm.chat(self.vlm_tokenizer, query=query, history=None)
        box = extract_box_coordinates(response)
        # self.vlm.cpu()
        return box

    def chat_with_llm(self, question):
        response = self.LLM.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": question},
            ],
            stream=False
        )
        return response.choices[0].message.content

    def split_total_des_to_subtasks(self, total_des):
        pass
    def get_target_from_total_des(self, total_des):
        user_prompt = '''
        我将给你一个机械臂操作的整体指令，这个指令是关于将某个物体放到另外一个物体上，请你依据这个指令，将其中的操作目标提取出来，并区分动静态（动态物体指将要被抓取的物体，静态物体指将要被放置的物体）
        然后将提取结果翻译成英语并按照此格式返回结果：(static obj name, dynamic obj name)
        例如，如果给定的整体指令为“把红色杯子放到蓝色盘子上”，则返回的结果应该是 (blue disk, red cup)
        注意：1.严格按照格式返回结果，不要有其他多余内容 2.静态物体在前，动态物体在后 3.翻译成简单但又没有歧义的英语单词
        给定的整体指令为：
        '''
        question = user_prompt + total_des
        response = self.chat_with_llm(question)
        try:
            target_static = response[1:-1].split(',')[0]
            target_dynamic = response[1:-1].split(',')[1]
            return target_static, target_dynamic
        except:
            return TARGET_EXTRACT_ERROR
        
    def get_target_from_total_des_ch(self, total_des):
        user_prompt = '''
        我将给你一个机械臂操作的整体指令，这个指令是关于将某个物体放到另外一个物体上，请你依据这个指令，将其中的操作目标提取出来，并区分动静态（动态物体指将要被抓取的物体，静态物体指将要被放置的物体）
        然后将提取结果按照此格式返回结果：(静态物体名称, 动态物体名称)
        例如，如果给定的整体指令为“把红色杯子放到蓝色盘子上”，则返回的结果应该是 (蓝色盘子, 红色杯子)
        注意：1.严格按照格式返回结果，不要有其他多余内容 2.静态物体在前，动态物体在后 3.提前出的物体名称要简单又没有歧义
        给定的整体指令为：
        '''
        user_prompt = '''
        我将给你一个机械臂操作的整体指令，请你依据这个指令，将其中的操作目标提取出来，并区分动静态（动态物体指将要被机械臂抓取的物体，静态物体指不需要移动的物体或者参照物）
        然后将提取结果按照此格式返回结果：(静态物体名称, 动态物体名称)
        例如，如果给定的整体指令为“把红色杯子放到蓝色盘子上”，则返回的结果应该是 (蓝色盘子, 红色杯子)
        注意：1.严格按照格式返回结果，不要有其他多余内容 2.静态物体在前，动态物体在后 3.提前出的物体名称要简单又没有歧义
        给定的整体指令为：
        '''

        question = user_prompt + total_des
        response = self.chat_with_llm(question)
        try:
            target_static = response[1:-1].split(',')[0]
            target_dynamic = response[1:-1].split(',')[1]
            return target_static, target_dynamic
        except:
            return TARGET_EXTRACT_ERROR
    def aug_total_des(self, total_des):
        user_prompt = '''
        我将给你一个机械臂操作的整体指令，这个指令是关于将某个物体放到另外一个物体上，请你对这个指令进行增广（换个说法但是保持含义不变），
        分别返回4个增广后的中文指令与英文指令，并返回结果，要求返回结果的格式如下（直接返回字符串形式，不要是json格式）：
        [
            [aug_zh1, aug_zh2, aug_zh3, aug_zh4],
            [aug_en1, aug_en2, aug_en3, aug_en4]
        ]
        指令为:
        '''
        question = user_prompt + total_des
        response = self.chat_with_llm(question)
        def parse_custom_string(custom_string):
            # 去除首尾的中括号和换行符
            cleaned_string = custom_string.strip().strip('[]').strip()
            
            # 分割字符串为各个子列表
            list_items = cleaned_string.split('],\n    [')
            
            # 去掉子列表的多余字符，并分割为各个元素
            parsed_data = []
            for item in list_items:
                # 去除前后的方括号和引号
                cleaned_item = item.strip().strip('[]').strip()
                
                # 分割字符串为元素列表，并去除两侧的引号
                elements = cleaned_item.split('", "')
                elements[0] = elements[0].lstrip('"')
                elements[-1] = elements[-1].rstrip('"')
                
                # 将元素列表加入主列表中
                parsed_data.append(elements)
    
            return parsed_data
        
        try:
            res = parse_custom_string(response)
        except:
            res = DES_AUG_ERROR
        return res

    def get_mask_from_box(self, cur_frame, box_npy):
        self.predictor.set_image(cur_frame, image_format="BGR")
        masks, _, _ = self.predictor.predict(box=box_npy)
        return masks[0] # (h, w)

    def detect(self, img_npy, text, box_threshold = 0.2, text_threshold = 0.25):
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

    def get_box(self, cur_frame, target_ls, model_type):
        # cur_frame: numpy array (H, W, 3), BGR mode
        # target_ls: list of str [red cup, green plate]
        h, w = cur_frame.shape[:2]
        res_boxes = []
        if model_type == 'dino':
            for target in target_ls:
                result = self.detect(cur_frame, target)
                scores = result[0]['scores'].cpu().numpy()
                if len(scores) > 0:
                    max_score_idx = scores.argmax()
                    box = result[0]['boxes'].cpu().numpy().astype(int).tolist()[max_score_idx]
                    res_boxes.append(box)
                else:
                    # return NO_DETECTION_ERROR
                    res_boxes.append([0,0,0,0])
        elif model_type =='vlm':
            # cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)
            tmp_img_path = 'tmp.jpg'
            cv2.imwrite(tmp_img_path, cur_frame)
            for target in target_ls:
                box = self.get_box_using_vlm(tmp_img_path, target)
                x1, y1, x2, y2 = (int(box[0] / 1000 * w), int(box[1] / 1000 * h), int(box[2] / 1000 * w), int(box[3] / 1000 * h))
                res_boxes.append([x1, y1, x2, y2])
            # box_image = drawbox(cv2.imread(tmp_img_path), res_boxes)
            # cv2.imwrite(f'box_{"_".join(target_ls)}.jpg', box_image)
            # self.free_gpu_memory()
        else:
            raise ValueError(f'unknown model_type: {model_type}. available options: dino, vlm')
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
        # mask_image = drawmask(cur_frame, res_masks)
        # cv2.imwrite(f'mask_{"_".join(target_ls)}.jpg', mask_image)
        return res_masks

    def sample_points_from_mask(self, mask, num):
        true_indices = np.argwhere(mask)
        if len(true_indices) < num:
            return true_indices.tolist()
        sampled_indices = true_indices[np.random.choice(len(true_indices), num, replace=False)]
        return sampled_indices.tolist()
    


    def get_mask_and_box_sam2_online(self, cur_frame, target_ls):
        original_device = torch.cuda.current_device()
        torch.cuda.set_device(1)
        res_masks = []
        if not self.start:
            # 通过grounding dino + mobilesam获取sam2的起始跟踪点（存在self.initial_points中）
            self.start = True
            boxes = self.get_box(cur_frame, target_ls, 'vlm')
            # assert len(boxes) == 2, f'len(boxes): {len(boxes)}'
            for box in boxes:
                mask = self.get_mask_from_box(cur_frame, np.array(box))
                self.initial_masks.append(mask)
                points = self.sample_points_from_mask(mask, 10)
                points = [[x[1], x[0]] for x in points]
                self.initial_points.append(points)

            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            # with torch.inference_mode():

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
                
                torch.cuda.set_device(original_device)
                return res_masks, compute_bounding_boxes(res_masks)
        
        else:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            # with torch.inference_mode():
                # 后续帧不再需要grounding dino与mobile sam，直接用sam2_predictor
                out_obj_ids, out_mask_logits = self.sam2_predictor.track(cur_frame)
                for i in range(len(out_mask_logits)):
                    res_masks.append(out_mask_logits[i][0].cpu().numpy() > 0)
                torch.cuda.set_device(original_device)
                return res_masks, compute_bounding_boxes(res_masks)

if __name__ == '__main__':
    uni_agent = UniAgent(sam_model_path='/x2robot/ganruyi/models/MobileSAM/mobile_sam.pt')

    # step1: 获取视频和整体描述
    total_des = '左臂把可乐放进米色盆子里'
    video_path = '/x2robot/zhengwei/10000/20240808-pick_up-item-night/20240808-pick_up-item-night@MASTER_SLAVE_MODE@2024_08_08_22_46_47/faceImg.mp4'
    video_path = '/home/ganruyi/code/faceImg.mp4'
    video_npy = video_2_npy(video_path)  # shape: (frames, h, w, 3)

    # step2: 调用LLM提取目标（如果解析出错会输出错误标志TARGET_EXTRACT_ERROR）
    target_ls = uni_agent.get_target_from_total_des(total_des)
    if target_ls == TARGET_EXTRACT_ERROR:
        print('target extract error')
    else:
        target_static, target_dynamic = target_ls
    # print(f'target_static:{target_static}, target_dynamic:{target_dynamic}')

    # step3: 检测 & 跟踪
    res_imgs_mask = []
    res_imgs_box = []
    for i in tqdm(range(video_npy.shape[0])):
        res_masks, res_boxes = uni_agent.get_mask_and_box_sam2_online(video_npy[i], target_ls)
        res_img_mask = visualize_masks_on_image(res_masks, video_npy[i])
        res_img_box = viz_boxes(video_npy[i], res_boxes, ['blue', 'red'])
        res_imgs_mask.append(res_img_mask)
        res_imgs_box.append(res_img_box)

    save_imgs_as_video(res_imgs_mask, 'res_mask.mp4')
    save_imgs_as_video(res_imgs_box, 'res_box.mp4')
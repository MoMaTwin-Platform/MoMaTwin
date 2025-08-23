import copy
from typing import Union

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

def annotate(image, static_result, dynamic_result) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for idx, label in enumerate(static_result['labels']):
        score = static_result['scores'][idx].cpu().numpy().tolist()
        box = static_result['boxes'][idx].cpu().numpy().astype(int).tolist()
        # cv2 is BGR, static is blue
        # color = np.array([255,0,0])
        # Draw bounding box
        cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
        # cv2.putText(image_cv2, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
    for idx, label in enumerate(dynamic_result['labels']):
        score = dynamic_result['scores'][idx].cpu().numpy().tolist()
        box = dynamic_result['boxes'][idx].cpu().numpy().astype(int).tolist()
        # cv2 is BGR, static is blue
        # color = np.array([0,0,255])
        # Draw bounding box
        cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
        # cv2.putText(image_cv2, f'{label}: {score:.2f}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
class GroundingDINOEncoder(ModuleAttrMixin):
    def __init__(self,
            name: str,
            frozen: bool=False,
            return_type: str='image', # image, embed
            box_threshold=0.3,
            text_threshold=0.3,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        """
        super().__init__()
        assert return_type in ['image', 'embed']
        self.return_type = return_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(name)

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, batched_img_tensor, batched_static_text, batched_dynamic_text):
        # print(f'batched_static_text: {batched_static_text}')
        # print(f'batched_dynamic_text: {batched_dynamic_text}')
        batched_static_text = [t+'.' for t in batched_static_text]
        batched_dynamic_text = [t+'.' for t in batched_dynamic_text]
        batched_target_size= [ [480,640] for _ in batched_static_text]
        
        static_inputs = self.processor(images=batched_img_tensor, text=batched_static_text, return_tensors="pt", padding='longest').to(self.device)
        dynamic_inputs = self.processor(images=batched_img_tensor, text=batched_dynamic_text, return_tensors="pt", padding='longest').to(self.device)
        with torch.no_grad():
            static_outputs = self.model(**static_inputs)
            dynamic_outputs = self.model(**dynamic_inputs)
        if self.return_type == 'image':
            static_results = self.processor.post_process_grounded_object_detection(
                static_outputs,
                static_inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=batched_target_size,
            )
            dynamic_results = self.processor.post_process_grounded_object_detection(
                dynamic_outputs,
                dynamic_inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=batched_target_size,
            )
            batched_img_arr = batched_img_tensor.cpu().numpy()
            batched_annotated_img = []
            for img,sr,dr in zip(batched_img_arr, static_results, dynamic_results):
                # print(f'annotated_img.shape before:{img.shape}')
                annotated_img = annotate(img, sr, dr)
                # print(f'annotated_img.shape 1:{annotated_img.shape}')
                batched_annotated_img.append(annotated_img)
            return torch.tensor(batched_annotated_img, device=self.device)
        elif self.return_type == 'embed':
            sh = static_outputs.last_hidden_state # B,Q(900),D(256)
            dh = dynamic_outputs.last_hidden_state # B,Q(900),D(256)
            pass
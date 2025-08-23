from typing import Union
import torch
from transformers import SiglipModel, SiglipTextModel, SiglipVisionModel, SiglipVisionConfig, SiglipTextConfig
from transformers import SiglipImageProcessor
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from torch import nn
from transformers import AutoTokenizer, SiglipTokenizer
import logging
logger = logging.getLogger(__name__)

def load_siglip_vision_model(name, pretrained=True):
    if pretrained:
        model = SiglipVisionModel.from_pretrained(name)
    else:
        config = SiglipVisionConfig()
        model = SiglipVisionModel(config)
    return model

def load_siglip_text_model(name, pretrained=True):
    if pretrained:
        model = SiglipTextModel.from_pretrained(name)
    else:
        config = SiglipTextConfig()
        model = SiglipTextModel(config)
    return model


class SiglipVisionEncoder(ModuleAttrMixin):
    def __init__(self, name, use_pooler=True, pretrained=True):
        super().__init__()
        self.vision_model = load_siglip_vision_model(name, pretrained)
        self.use_pooler = use_pooler
        self.config = self.vision_model.config
    
    """
    Assumes x shape: B*T,C,H,W
    """
    def forward(self, x):
        # do image transform at multi_image_obs_encoder, do not use transformers' image_processor
        output = self.vision_model(x)
        pooler_output = torch.unsqueeze(output.pooler_output, dim=1) if self.use_pooler else output.last_hidden_state
        return pooler_output
    
    @torch.no_grad()
    def output_shape(self):
        hidden_size = self.config.hidden_size
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        num_of_patch = (image_size // patch_size)**2
        if self.use_pooler:
            return (1, hidden_size)
        else:
            return (num_of_patch, hidden_size)
   

class SiglipTextEncoder(ModuleAttrMixin):
    def __init__(self, name, use_pooler=True, pretrained=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.text_model = load_siglip_text_model(name, pretrained)
        self.use_pooler = use_pooler
        logger.info(
            "SiglipTextEncoder number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    """
    Assumes Text input: B,L
    """
    def forward(self, instruction):
        # print(type(instruction), instruction)
        # inputs = self.tokenizer(instruction, padding='max_length', return_tensors='pt').to(self.device)
        inputs = self.tokenizer(instruction, padding='longest', return_tensors='pt').to(self.device)
        # print(instruction, inputs)
        output = self.text_model(**inputs)
        pooler_output = torch.unsqueeze(output.pooler_output, dim=1) if self.use_pooler else output.last_hidden_state
        return pooler_output
    
    @torch.no_grad()
    def output_shape(self):
        example_instruct = ['Place the brown cup onto the glass plate.']
        example_output = self.forward(example_instruct)
        output_shape = example_output.shape[1:]
        return output_shape


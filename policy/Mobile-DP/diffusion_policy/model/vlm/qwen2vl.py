#coding=utf8
import torch
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusion_policy.model.vlm.qwen_vl_utils import process_vision_info 
from peft import PeftModel, LoraConfig, get_peft_model
import logging
logger = logging.getLogger(__name__)

class Qwen2VLEncoder(ModuleAttrMixin):
    def __init__(self, name, shape_meta: dict, use_pooler='mean', freeze='none'):
        super().__init__()
        use_pooler = use_pooler.lower()
        assert use_pooler in ['mean', 'max']
        self.use_pooler = use_pooler
        self.shape_meta = shape_meta
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            name,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            )
        # self.model = self.model.model
        print(f'self.device:{self.device}', flush=True)
        if freeze == 'lora':
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        # self.model.to('cuda')
        logger.info(
            "Qwen2VLEncoder number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(name)
        self.prompt = 'Find all objects mentioned in the instruction.'

    """
    obs_dict: dict, every value for key shape is: B,T,H,W,C 
    instruction: B,T
    """
    def forward(self, obs_dict, instructions):
        messages = []
        content = []
        for _ in obs_dict.items():
            content.append({"type": "image"})
        batched_image = []
        for idx, instruction in enumerate(instructions):
            new_content = content.copy()
            new_content.append({"type": "text", "text": f"Instruction: {instruction}\n{self.prompt}"})
            message = [
                {
                    "role": "user",
                    "content": new_content
                },
            ]
            messages.append(message)
            images = []
            for obs_key,obs_value in obs_dict.items(): # obs_value: B,T,H,W,C
                for img in obs_value[idx]:
                    images.append(img)
            batched_image.append(images)

        # Preparation for inference
        batched_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=batched_text,
            images=batched_image,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        output = self.model(**inputs, output_hidden_states=True, return_dict=True) # batch_size, num_token, dim
        last_hidden_state = output.hidden_states[-1]
        if self.use_pooler == 'mean':
            output = last_hidden_state.mean(dim=1)
        elif self.use_pooler == 'max':
            output = last_hidden_state.max(dim=1)
        return output

    @torch.no_grad()
    def output_shape(self):
        return (self.model.config.hidden_size,)
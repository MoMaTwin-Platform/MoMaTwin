from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
logger = logging.getLogger(__name__)


class DistilBertEncoder(ModuleAttrMixin):
    def __init__(self, name, max_len, use_cls=True, sum_pooling=False, freeze='none'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.model = AutoModel.from_pretrained(name)
        self.max_len = max_len
        self.use_cls = use_cls
        self.sum_pooling = sum_pooling
        assert sum([self.use_cls, self.sum_pooling]) == 1
        freeze = freeze.lower()
        if freeze == 'all':
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze.startswith('all_ex_last_'):
            # 解析需要解冻的层数
            parts = freeze.split('all_ex_last_')
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(
                    f"Invalid freeze format: {freeze}. "
                    "Expected format 'all_ex_last_x' where x is an integer."
                )
            unfreeze_layers = int(parts[1])
            for name, param in self.model.named_parameters():
                if not name.startswith('transformer.layer'):
                    param.requires_grad = False
                else:
                    layer_num = int(name.split('.')[2])
                    total_layers = len(self.model.transformer.layer)
                    if layer_num < total_layers - unfreeze_layers:
                        param.requires_grad = False
        elif freeze == 'none':
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("all params is trainable")
        else:
            raise ValueError(f"unknown freeze type：{self.freeze}. not in 'all', 'all_ex_last', 'none'.")

        logger.info(
            "number of trainable parameters: %e", sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def forward(self, instruction):
        # process instruction string
        input_ids = self.tokenizer(instruction, max_length=self.max_len, padding='longest', return_tensors='pt').to(self.device)
        
        # 调试信息：打印instruction和input_ids
        if hasattr(self, '_debug_print_count'):
            self._debug_print_count += 1
        else:
            self._debug_print_count = 1
        
        # 只在前几次调用时打印，避免日志过多
        if self._debug_print_count <= 3:
            print(f"[DistilBert Debug {self._debug_print_count}]")
            print(f"Input instruction: {instruction}")
            print(f"Input_ids shape: {input_ids['input_ids'].shape}")
            print(f"Input_ids: {input_ids['input_ids']}")
            
            # 解码回文本验证
            decoded_text = self.tokenizer.batch_decode(input_ids['input_ids'], skip_special_tokens=True)
            print(f"Decoded text: {decoded_text}")
            print("-" * 50)
        
        last_hidden = self.model(**input_ids).last_hidden_state
        output_hidden = None
        if self.use_cls:
            # B,D
            output_hidden = last_hidden[:,0,:].squeeze(1)
        if self.sum_pooling:
            # B,D
            output_hidden = last_hidden.sum(1).squeeze(1)
        # print(output_hidden.shape)
        return output_hidden
    
    @torch.no_grad()
    def output_shape(self):
        # batch_size = 1
        # example_instruct = torch.zeros((batch_size, ), dtype=self.dtype, device=self.device)
        example_instruct = ["pickup the orange"]
        # print(example_obs_dict)
        example_output = self.forward(example_instruct)
        output_shape = example_output.shape[1:]
        return output_shape

if __name__ == '__main__':
    logger.setLevel('INFO')
    # 添加 StreamHandler 以确保日志信息被输出
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    name = '/x2robot/ganruyi/models/distilbert-base-uncased'
    max_len = 40
    use_cls = True
    model = DistilBertEncoder(name=name, max_len=max_len, use_cls=use_cls, freeze='all_ex_last_2')
    # model = DistilBertEncoder(name=name, max_len=max_len, use_cls=use_cls, freeze='none')
    print(model.output_shape)

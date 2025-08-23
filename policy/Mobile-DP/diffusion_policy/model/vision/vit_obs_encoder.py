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

logger = logging.getLogger(__name__)

class VitObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: nn.Module,
            transforms: Union[list[nn.Module], nn.Module],
            frozen: bool=False,
            pretrained: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict() 

        if frozen:
            assert pretrained, 'Need a pretrained model to be frozen'
            for param in rgb_model.parameters():
                param.requires_grad = False


        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
    
        if transforms is None:
            transform = nn.Identity()
        else:
            transform = transforms

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = rgb_model if share_rgb_model else copy.deepcopy(rgb_model)
                key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, obs_dict):
        features = list()
        # batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        # for key in self.rgb_keys:
        #     img = obs_dict[key]
        #     B, T = img.shape[:2]
        #     assert B == batch_size
        #     # assert img.shape[2:] == self.key_shape_map[key]
        #     img = img.reshape(B*T, *img.shape[2:])
        #     print(f'img.shape: {img.shape}, B: {B}, T: {T}')
        #     img = self.key_transform_map[key](img)
        #     # print(f'img.shape[2:] == self.key_shape_map[key]:', img.shape[2:], self.key_shape_map[key])
        #     # assert img.shape[2:] == self.key_shape_map[key]
        #     raw_feature = self.key_model_map[key](img)
        #     # vit already uses the CLS token
        #     feature = raw_feature
        #     assert len(feature.shape) == 2 and feature.shape[0] == B * T
        #     features.append(feature.reshape(B, -1))
        batch_size = None
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                # print('1 img shape:', img.shape)
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map[self.rgb_keys[0]](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        # # process lowdim input
        # for key in self.low_dim_keys:
        #     data = obs_dict[key]
        #     B, T = data.shape[:2]
        #     assert B == batch_size
        #     print(data.shape, data.shape[2:], self.key_shape_map[key])
        #     assert data.shape[2:] == self.key_shape_map[key]
        #     features.append(data.reshape(B, -1))
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # print(data.shape[1:], self.key_shape_map[key])
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            # shape = (224,224)
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        # print(example_obs_dict)
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

    # @torch.no_grad()
    # def output_shape(self):
    #     example_obs_dict = dict()
    #     obs_shape_meta = self.shape_meta['obs']
    #     for key, attr in obs_shape_meta.items():
    #         shape = tuple(attr['shape'])
    #         this_obs = torch.zeros(
    #             (1, attr['horizon']) + shape, 
    #             dtype=self.dtype,
    #             device=self.device)
    #         example_obs_dict[key] = this_obs
    #     example_output = self.forward(example_obs_dict)
    #     assert len(example_output.shape) == 2
    #     assert example_output.shape[0] == 1
        
    #     return example_output.shape

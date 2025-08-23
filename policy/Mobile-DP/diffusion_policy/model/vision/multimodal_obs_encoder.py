import math
from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.randomizer import ColorRandomizer, CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import logging
logger = logging.getLogger(__name__)

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    @torch.no_grad
    def forward(self, x):
        # return torch.clamp(x.float()*torch.tensor(0.00392156862745098), 0, 1.0)
        return torch.clamp(x.float()/255.0, 0.0, 1.0)
    
class MultiModalObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            lm_encoder: nn.Module = None,
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            norm_mean = [0.485, 0.456, 0.406],
            norm_std = [0.229, 0.224, 0.225],
            imagenet_norm: bool=False,
            color_randomizer_prob = 0,
            # whether div 255
            div255: bool=False,
            pretrained: bool=False,
            feature_aggregation = None,
            feature_aggregation_dim = 768,
            feature_aggregation_pe = 'learned',
            feature_aggregation_layers = 4,
            last_hidden_aggregation = 'avg',
            freeze_lm_encoder = False,
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

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model
            if use_group_norm:
                print('replace_submodules BatchNorm2d to GroupNorm')
                rgb_model = replace_submodules(
                    root_module=rgb_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features//16, 
                        num_channels=x.num_features)
                )
                key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        print('replace_submodules BatchNorm2d to GroupNorm')
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure image process, always: crop_randomizer(do center-crop on eval)->resize->totensor->color_randomizer(optional, do nothing on eval)->Normalize
                
                # input_shape = shape
                crop_randomizer = nn.Identity()
                assert crop_shape is not None
                if random_crop:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape                    
                    crop_randomizer = CropRandomizer(input_shape=shape, crop_height=h, crop_width=w)
                    # input_shape = (shape[0], h, w)
                else:
                    h, w = crop_shape 
                    crop_randomizer = torchvision.transforms.CenterCrop((h,w))
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w),
                        interpolation = torchvision.transforms.InterpolationMode("bicubic"),
                        antialias=True
                    )
                # configure randomizer
                color_randomizer = nn.Identity()
                if color_randomizer_prob > 0:
                    color_randomizer = ColorRandomizer(input_shape=(shape[0], resize_shape[0], resize_shape[1]), brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                    # color_randomizer = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
                    color_randomizer = torchvision.transforms.RandomApply(torch.nn.ModuleList([color_randomizer]), p=color_randomizer_prob)
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=torch.tensor(norm_mean), std=torch.tensor(norm_std))
                if div255:
                    div255_normalizer = Normalize()
                    if color_randomizer_prob > 0:
                        this_transform = nn.Sequential(crop_randomizer, this_resizer, div255_normalizer, color_randomizer, this_normalizer)
                    else:
                        this_transform = nn.Sequential(crop_randomizer, this_resizer, div255_normalizer, this_normalizer)
                else:
                    assert color_randomizer_prob == 0 # color_randomizer must be with div255 
                    this_transform = nn.Sequential(crop_randomizer, this_resizer, this_normalizer)
                key_transform_map[key] = this_transform
                # print(f'color_randomizer_prob:{color_randomizer_prob}, {key}:\n', this_transform)
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.feature_aggregation = feature_aggregation
        self.last_hidden_aggregation = last_hidden_aggregation
        # get feature dim
        if feature_aggregation is not None and lm_encoder is not None:
            if self.last_hidden_aggregation == 'soft_attention':
                self.attention = nn.Sequential(
                    nn.Linear(feature_aggregation_dim, 1, bias=False),
                    nn.Softmax(dim=1)
                )
            obs_feature_shape = key_model_map['rgb'].output_shape()
            lm_feature_shape = lm_encoder.output_shape()
            assert len(obs_feature_shape) == 2 and len(lm_feature_shape) == 2 and obs_feature_shape[-1] == lm_feature_shape[-1] and lm_feature_shape[-1] == feature_aggregation_dim, 'obs_feature_shape and lm_feature_shape must be L,D and D is the same as feature_aggregation_dim'
            if self.feature_aggregation == 'transformer':
                if feature_aggregation_pe == 'learnable':
                    self.position_embedding = torch.nn.Parameter(torch.randn(obs_feature_shape[0] + lm_feature_shape[0] + 1, feature_aggregation_dim))
                elif feature_aggregation_pe == 'sinusoidal':
                    num_features = obs_feature_shape[0] + lm_feature_shape[0] + 1
                    self.position_embedding = torch.zeros(num_features, feature_aggregation_dim)
                    position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, feature_aggregation_dim, 2).float() * (-math.log(2 * num_features) / feature_aggregation_dim))
                    self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                    self.position_embedding[:, 1::2] = torch.cos(position * div_term)
                else:
                    assert f'{feature_aggregation_pe} is not implemented!'
                self.aggregation_transformer = nn.TransformerEncoder(
                    encoder_layer=nn.TransformerEncoderLayer(d_model=feature_aggregation_dim, nhead=4, batch_first=True, norm_first=True),
                    num_layers=feature_aggregation_layers)
        else:
            self.aggregation_transformer = None
            # if lm_encoder is not None, check instruction 
            # if lm_encoder is not None:
            #     lm_feature_dim = lm_encoder.output_shape()[-1]
            #     obs_feature_dim += lm_feature_dim
            #     print(f'after add: obs_feature_dim:{obs_feature_dim}, lm_feature_dim:{lm_feature_dim}')

        self.lm_encoder = lm_encoder
        if freeze_lm_encoder:
            self.lm_encoder.requires_grad_(False)
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.div255 = div255

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "number of trainable parameters: %e, total parameters: %e",
            trainable_params,
            total_params,
        )


    def forward(self, obs_dict, instructions):
        # instructions shape [B]
        batch_size = obs_dict['agent_pos'].shape[0]
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                # B,T,H,W,C -> B,T,C,H,W
                if img.shape[2] != 3 and img.shape[-1] == 3:
                    img = img.moveaxis(-1,2)

                B, T = img.shape[:2]
                assert B == batch_size
                # assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B*T, *img.shape[2:])
                # print(f' before: {B},{T},{img.shape}')
                img = self.key_transform_map[key](img)
                # print(f' after: {B},{T},{img.shape}')
                # print('2 img shape:', img.shape)
                imgs.append(img)
            # (N*B*T,C,H,W), N is for #view
            imgs = torch.cat(imgs, dim=0)
            # (N*B*T,D) or (N*B*T, L, D) L is for #patch or #token
            feature = self.key_model_map['rgb'](imgs)
            # (N*T,B,D) or (N*T,B,L,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N*T,D) or (B,N*T,L,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*T,D) or (B,N*T*Li,D) # Li for #patch token
            rgb_feature = feature.reshape(batch_size,-1,feature.shape[-1])
            features.append(rgb_feature)
        else:
            # run each rgb obs to independent models
            assert False, 'TODO: share_rgb_model=False'
            for key in self.rgb_keys:
                img = obs_dict[key]
                B, T = img.shape[:2]
                assert B == batch_size
                img = img.reshape(B*T, *img.shape[2:])
                img = self.key_transform_map[key](img)
                rgb_feature = self.key_model_map[key](img)
                features.append(rgb_feature)
        # process lm input
        if instructions is not None and self.lm_encoder is not None:
            lm_feature = self.lm_encoder(instructions) # (B,1,D) or (B,Ll,D), Ll for #token
            features.append(lm_feature)

        # process feature_aggregation
        if self.feature_aggregation == 'transformer':
            hidden = torch.concat(features, dim=1) # (B,N*T*Li+Ll,D)
            hidden = self.aggregation_transformer(hidden) # (B,N*T*Li+Ll,D)
            if self.last_hidden_aggregation == 'avg':
                hidden = torch.mean(hidden, dim=[1], keepdim=True) # (B,1,D)
            elif self.last_hidden_aggregation == 'max':
                hidden = torch.amax(hidden, dim=[1], keepdim=True)
            elif self.last_hidden_aggregation == 'soft_attention':
                weight = self.attention(hidden)
                hidden = torch.sum(hidden * weight, dim=1, keepdim=True)
            elif self.last_hidden_aggregation == 'spatial_embedding':
                hidden = torch.mean(hidden * self.spatial_embedding, dim=1, keepdim=True)
            else:
                assert self.feature_aggregation is None, 'must have a last_hidden_aggregation type in [avg, max, soft_attention]'
            
            result = hidden[:,0,:] # (B,D)
            # process lowdim input
            for key in self.low_dim_keys:
                data = obs_dict[key]
                # assert data.shape[2:] == self.key_shape_map[key]
                data = data.reshape(batch_size, -1)
                result = torch.cat([result, data], dim=-1)
        else:
            for key in self.low_dim_keys:
                data = obs_dict[key]
                # assert data.shape[2:] == self.key_shape_map[key]
                data = data.reshape(batch_size, -1)
                features.append(data)
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
                (batch_size, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        # print(example_obs_dict)
        instructions = ['pickup the cup onto the plate']
        example_output = self.forward(example_obs_dict, instructions)
        output_shape = example_output.shape[1:]
        return output_shape

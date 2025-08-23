from typing import Dict, Sequence, Tuple, Union
import copy
from diffusion_policy.model.vision.spatial_softmax import SpatialSoftmax
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.randomizer import ColorRandomizer, CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.groundingdino import GroundingDINOEncoder
import logging
logger = logging.getLogger(__name__)

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
    @torch.no_grad
    def forward(self, x):
        # return torch.clamp(x.float()*torch.tensor(0.00392156862745098), 0, 1.0)
        return torch.clamp(x.float()/255.0, 0.0, 1.0)
    
class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
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
            spatial_softmax_num_keypoints: int=None, # use spatial_softmax, usually 32
            global_pool_dim_out: int=None, # Output dimension for the global pooling operation (a projection is applied prior to pooling).
            use_img_instruction: bool=False, # whether use img instruction
            grounding_rgb_keys:Union[str, Sequence[str], None]=None, # grounding which rgb_key by grounding dino
            grounding_model: GroundingDINOEncoder=None,
            mask_type=None,
            lighting_augmentation=None,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        tactile_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        if mask_type:
            assert mask_type in ['mask_only', 'mask_as_channel'], f'{mask_type} not supported!!!'        

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            if spatial_softmax_num_keypoints: # if use spatial_softmax, drop resnet avg_pool and fc layer. only for resnet!!!
                rgb_model = nn.Sequential(*(list(rgb_model.children())[:-2]))
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
        if grounding_rgb_keys is not None:
            self.grounding_rgb_keys = set(grounding_rgb_keys)
            self.grounding_model = grounding_model.to(self.device)
        else:
            self.grounding_rgb_keys = None
    
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
            elif type == 'tactile':
                tactile_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        if mask_type == 'mask_only':
            h, w = resize_shape
            this_resizer = torchvision.transforms.Resize(
                size=(h,w),
                interpolation = torchvision.transforms.InterpolationMode("bicubic"),
                antialias=True
            )
            if imagenet_norm:
                this_normalizer = torchvision.transforms.Normalize(
                    mean=torch.tensor(norm_mean), std=torch.tensor(norm_std))
            if div255:
                div255_normalizer = Normalize()
                if color_randomizer_prob > 0:
                    this_transform = nn.Sequential(this_resizer, div255_normalizer, color_randomizer, this_normalizer)
                else:
                    this_transform = nn.Sequential(this_resizer, div255_normalizer, this_normalizer)
            # self.img_instruction_transform = this_transform
            self.img_instruction_transform = key_transform_map['face_view']
        
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        self.mask_type = mask_type
        self.use_img_instruction = use_img_instruction
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.div255 = div255
        self.pool = None
        self.spatial_softmax_num_keypoints = spatial_softmax_num_keypoints
        if spatial_softmax_num_keypoints:
            # for resnet50
            self.ss = SpatialSoftmax(input_shape=(2048,7,7), num_kp=spatial_softmax_num_keypoints)
            self.pool_linear = nn.Linear(spatial_softmax_num_keypoints*2, spatial_softmax_num_keypoints*2)
            self.pool_relu = nn.ReLU()
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward_img_instruction(self, img_instruction):
        batch_size = None
        img = img_instruction
        # B,T,H,W,C -> B,T,C,H,W
        if img.shape[2] != 3 and img.shape[-1] == 3:
            img = img.moveaxis(-1,2)
        # print('1 img shape:', img.shape)
        if batch_size is None:
            batch_size = img.shape[0]
        else:
            assert batch_size == img.shape[0]

        B, T = img.shape[:2]
        assert B == batch_size
        # assert img.shape[2:] == self.key_shape_map[key]
        img = img.reshape(B*T, *img.shape[2:])
        # print(f' before: {B},{T},{img.shape}')
        # use a default transform for temp, only use center crop todo: config
        img = self.img_instruction_transform(img) 
        # print(f' after: {B},{T},{img.shape}')
        # (B*T,C,H,W)
        imgs = img
        # (N*B*T,D)
        feature = self.key_model_map['rgb'](imgs)
        # print(f'feature: {feature.shape}')
        if self.spatial_softmax_num_keypoints:
            feature = torch.flatten(self.ss(feature), start_dim=1)
            # print(f'feature shape 1: {feature.shape}')
            feature = self.pool_linear(feature)
            # print(f'feature shape 2: {feature.shape}')
            feature = self.pool_relu(feature)
            # print(f'feature shape 3: {feature.shape}')
        # (N*T,B,D)
        feature = feature.reshape(-1,batch_size,*feature.shape[1:])
        # (B,N*T,D)
        feature = torch.moveaxis(feature,0,1)
        # (B,N*T*D)
        feature = feature.reshape(batch_size,-1)
        return feature

    def forward(self, obs_dict, batched_static_text=None, batched_dynamic_text=None, batched_mask=None):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if self.grounding_rgb_keys is not None and key in self.grounding_rgb_keys:
                    if img.shape[-1] not in [3,5]: # 只有图片是3，如果mask加入到图片的channel里就是5
                        img = img.moveaxis(2,-1)
                    b,t,h,w,c = img.shape
                    # print(f'before multi img.shape: {img.shape}')
                    img = img.reshape(b*t,h,w,c)
                    img = self.grounding_model.forward(batched_img_tensor=img, batched_static_text=batched_static_text, batched_dynamic_text=batched_dynamic_text)
                    # print(f'after multi img.shape: {img.shape}')
                    img = img.reshape(b,t,h,w,c)
                # B,T,H,W,C -> B,T,C,H,W
                if img.shape[2] not in [3,5] and img.shape[-1] in [3,5]:
                    img = img.moveaxis(-1,2)
                # print('1 img shape:', img.shape)
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]

                B, T = img.shape[:2]
                assert B == batch_size
                # assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B*T, *img.shape[2:])
                # print(f' before: {B},{T},{img.shape}')
                img = self.key_transform_map[key](img)
                # print(f' after: {B},{T},{img.shape}')
                # print('2 img shape:', img.shape)
                imgs.append(img)
            # (N*B*T,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B*T,D)
            feature = self.key_model_map['rgb'](imgs)
            if self.spatial_softmax_num_keypoints:
                feature = torch.flatten(self.ss(feature), start_dim=1)
                # print(f'feature shape 1: {feature.shape}')
                feature = self.pool_linear(feature)
                # print(f'feature shape 2: {feature.shape}')
                feature = self.pool_relu(feature)
                # print(f'feature shape 3: {feature.shape}')
            # (N*T,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N*T,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*T*D)
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
                B, T = img.shape[:2]
                assert B == batch_size
                # assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B*T, *img.shape[2:])
                # debug face_view: torch.Size([1, 3, 480, 640]), (3, 480, 640)
                # print(f'debug {key}: {img.shape[1:]}, {self.key_shape_map[key]}')
                # assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        batch_size = None # reset again, because the lowdim is not the same obs horizon with image
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # print(f'obs lowdim {key}: {data.shape}, {self.key_shape_map[key]}')
            assert data.shape[2:] == self.key_shape_map[key], f'obs lowdim {key}: {data.shape}, {self.key_shape_map[key]}'
            data = data.reshape(batch_size, -1)
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
            obs_type = tuple(attr['type'])
            if self.mask_type == 'mask_as_channel' and obs_type == 'rgb':
                shape = (5, shape[1], shape[2]) # shape原来应该是(3,h,w)
            # shape = (224,224)
            this_obs = torch.zeros(
                (batch_size, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        # print(example_obs_dict)
        batched_static_text = None
        batched_dynamic_text = None
        if self.grounding_rgb_keys is not None:
            batched_static_text = ['plate']
            batched_dynamic_text = ['cup']
        example_output = self.forward(example_obs_dict,batched_static_text,batched_dynamic_text)
        output_shape = example_output.shape[1:]
        return output_shape
    
    @torch.no_grad()
    def output_img_instruct_shape(self):
        batch_size = 1
        obs_shape_meta = self.shape_meta['obs']
        shape = obs_shape_meta['face_view'].shape[1:] # first dim is C
        # h,w = self.resize_shape
        h,w = self.resize_shape if not self.random_crop else shape
        img_instruction = torch.zeros((batch_size, 1,h,w,3), dtype=self.dtype, device=self.device)
        example_output = self.forward_img_instruction(img_instruction)
        output_shape = example_output.shape[1:]
        return output_shape


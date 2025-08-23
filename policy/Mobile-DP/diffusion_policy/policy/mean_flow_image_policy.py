from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d_timezone import ConditionalUnet1DTimeZone
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from x2robot_dataset.common.data_utils import decode_text
from einops import rearrange
from functools import partial
import numpy as np

class MeanFlowPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDIMScheduler,
            obs_encoder: MultiImageObsEncoder,
            lm_encoder: nn.Module = None,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            mask_type=None,
            mask_keys=None,
            jvp_api='autograd',
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # n_obs_steps = shape_meta['obs']['agent_pos']['horizon']
        # get feature dim
        self.rgb_keys = set()
        self.lowdim_keys = set()
        self.tactile_keys = set()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                self.rgb_keys.add(key)
            elif type == 'low_dim':
                self.lowdim_keys.add(key)
            elif type == 'tactile':
                self.tactile_keys.add(key)
        obs_feature_dim = obs_encoder.output_shape()[0]
        print(f'obs_feature_dim:{obs_feature_dim}')
        self.norm_ignore_keys = self.rgb_keys
        # if lm_encoder is not None, check instruction 
        if lm_encoder is not None:
            lm_feature_dim = lm_encoder.output_shape()[0]
            obs_feature_dim += lm_feature_dim
            print(f'after add: obs_feature_dim:{obs_feature_dim}, lm_feature_dim:{lm_feature_dim}')
        self.use_tactile = len(self.tactile_keys) > 0
        if self.use_tactile:
            # 添加触觉编码器
            self.tactile_encoder = nn.Sequential(
                # 左右夹爪各60维,共120维
                nn.Linear(120, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32)  # 压缩到64维用于融合
            ) 
            obs_feature_dim += 32
            print(f'after add tactile encoder: obs_feature_dim:{obs_feature_dim}, tactile_feature_dim:{32}')
        self.mask_type = mask_type
        self.mask_keys = mask_keys
        if mask_type == 'mask_only' and mask_keys and len(mask_keys) > 0:
            before_dim = obs_feature_dim
            obs_feature_dim += obs_encoder.output_img_instruct_shape()[0] * len(mask_keys) 
            print(f'after add: obs_feature_dim:{obs_feature_dim}, before_dim:{before_dim}')

        # create diffusion model
        assert obs_as_global_cond, 'assert obs_as_global_cond'

        input_dim = action_dim
        # global_cond_dim = obs_feature_dim * n_obs_steps
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1DTimeZone(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        

        self.obs_encoder = obs_encoder
        self.lm_encoder = lm_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training

        # self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        self.flow_ratio=0.50

    def stopgrad(self, x):
        return x.detach()

    def sample_t_r(self, batch_size, device):
        samples = np.random.rand(batch_size, 2).astype(np.float32)

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def register_visual_instruct(self, visual_instructor):
        self.visual_instructor = visual_instructor

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        c = global_cond
        z = trajectory
        t = torch.ones((c.shape[0],), device=c.device)
        r = torch.zeros((c.shape[0],), device=c.device)
        z = z - model(z, t, r, global_cond=c)
        trajectory = z

        return trajectory

    def predict_action(self, batch) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        instructions: language instruction
        img_instructions: image instruction
        """

        # if in trainning mode and self has visual_instructor, use it to modify input observation
        # we do not modify input observation in inference time because the modification will be done outside the model
        if torch.is_grad_enabled() and hasattr(self, 'visual_instructor'):
            batch = self.visual_instructor.modify_obs(batch)

        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'], ignore_keys=self.rgb_keys)

        # nobs = self.normalizer.normalize(batch['obs'])
        # nobs = batch['obs']
        B = nobs['agent_pos'].shape[0]
        T = self.action_horizon
        Da = self.action_dim
        Do = self.obs_feature_dim

        # 简化instruction处理 - 现在直接是字符串列表
        instructions = batch.get('instruction', None)

        global_cond = None
        # B,T,C,H,W - 不再需要ignore instruction
        this_nobs = nobs
        batched_static_text = None
        batched_dynamic_text = None
        if 'static_text' in batch and 'dynamic_text' in batch:
            batched_static_text = batch['static_text']
            batched_dynamic_text = batch['dynamic_text']        
        # build input
        device = self.device
        dtype = self.dtype
        # condition through global feature
        nobs_features = self.obs_encoder(this_nobs, batched_static_text, batched_dynamic_text)
        if self.mask_type == 'mask_only' and self.mask_keys and len(self.mask_keys) > 0:
            for mask_key in self.mask_keys:
                img_instruction = batch[mask_key]
                img_instruct_features = self.obs_encoder.forward_img_instruction(img_instruction)
                nobs_features = torch.concat([nobs_features, img_instruct_features], dim=-1)
        if self.lm_encoder is not None and instructions is not None:
            this_instruct = instructions
            nlm_features = self.lm_encoder(this_instruct)
            nobs_features = torch.concat([nobs_features, nlm_features], dim=-1)
        if self.use_tactile:
            tactile_data = batch['tactile']
            tactile_data_normalized = self.normalizer['tactile'].normalize(tactile_data)
            # First encode the tactile data
            tactile_data_features = self.tactile_encoder(tactile_data_normalized)  # [B, T, 32]
            # Average pool over the time dimension
            tactile_data_features = torch.mean(tactile_data_features, dim=1)  # [B, 32]
            nobs_features = torch.concat([nobs_features, tactile_data_features], dim=-1)
        
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, T, Da)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_step_forward(self, batch):
        # if self has visual_instructor, use it to modify input observation
        if hasattr(self, 'visual_instructor'):
            batch = self.visual_instructor.modify_obs(batch)

        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'], ignore_keys=self.rgb_keys)
        # nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # 简化instruction处理 - 现在直接是字符串列表
        instructions = batch.get('instruction', None)

        batch_size = nactions.shape[0]
        global_cond = None
        trajectory = nactions
        # B,T,C,H,W - 不再需要ignore instruction
        this_nobs = nobs
        batched_static_text = None
        batched_dynamic_text = None
        if 'static_text' in batch and 'dynamic_text' in batch:
            batched_static_text = batch['static_text']
            batched_dynamic_text = batch['dynamic_text']
        nobs_features = self.obs_encoder(this_nobs, batched_static_text, batched_dynamic_text)
        if self.lm_encoder is not None:
            # this_instruct = self.lm_encoder(instructions).squeeze(1).repeat([1, To, 1]).reshape(batch_size*To, -1)
            nlm_features = self.lm_encoder(instructions)
            # print(f'nobs_features shape: {nobs_features.shape}, nlm_features: {nlm_features.shape}')
            nobs_features = torch.concat([nobs_features, nlm_features], dim=-1)
        if self.use_tactile:
            tactile_data = batch['tactile']
            tactile_data_normalized = self.normalizer['tactile'].normalize(tactile_data)
            # First encode the tactile data
            tactile_data_features = self.tactile_encoder(tactile_data_normalized)  # [B, T, 32]
            # Average pool over the time dimension
            tactile_data_features = torch.mean(tactile_data_features, dim=1)  # [B, 32]
            nobs_features = torch.concat([nobs_features, tactile_data_features], dim=-1)

        if self.mask_type == 'mask_only' and self.mask_keys and len(self.mask_keys) > 0:
            for mask_key in self.mask_keys:
                img_instruction = batch[mask_key]
                img_instruct_features = self.obs_encoder.forward_img_instruction(img_instruction)
                nobs_features = torch.concat([nobs_features, img_instruct_features], dim=-1)
        # print(f'nobs_features.shape: {nobs_features.shape}')
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)


        '''
        # fn(z, r, t): function to predict u
        # x: training batch
        t, r = sample_t_r()
        e = randn_like(x)
        z = (1 - t) * x + t * e
        v = e - x
        u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
        u_tgt = v - (t - r) * dudt
        error = u - stopgrad(u_tgt)
        loss = metric(error)
        '''

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)  # [B, T, dim]
        
        # sample two random timesteps ( t > r )
        t, r = self.sample_t_r(batch_size, trajectory.device)   
        t_ = rearrange(t, "b -> b 1 1 ")
        r_ = rearrange(r, "b -> b 1 1 ")

        # contstruct a waypoint z using x, e, t_
        z = (1 - t_) * trajectory + t_ * noise

        # compute velocity
        v = noise - trajectory
        v_hat = v

        model_partial = partial(self.model, global_cond=global_cond)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        pred = u
        target = self.stopgrad(u_tgt)

        return pred, target

    def forward(self, batch, predict=False):
        if predict:
            return self.predict_action(batch)
        else:
            return self.train_step_forward(batch)


import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
class DiffusionUnetMultiModalPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDIMScheduler,
            obs_encoder: nn.Module,
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
        obs_shape_meta = shape_meta['obs']
        agent_pos_dim = 0 
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            horizon = attr.get('horizon', 1)
            shape = attr.get('shape', [14])
            if type == 'rgb':
                self.rgb_keys.add(key)
            elif type == 'low_dim':
                self.lowdim_keys.add(key)
                assert len(shape) == 1
                agent_pos_dim += shape[0]*horizon 

        self.mask_type = mask_type
        self.mask_keys = mask_keys
        if mask_type == 'mask_only' and mask_keys and len(mask_keys) > 0:
            before_dim = obs_feature_dim
            # obs_feature_dim += obs_encoder.output_img_instruct_shape()[0] * len(mask_keys) 
            print(f'after add: obs_feature_dim:{obs_feature_dim}, before_dim:{before_dim}')
        print(f'agent_pos dim:{agent_pos_dim}', flush=True)
        obs_feature_dim = obs_encoder.output_shape()[0] + agent_pos_dim

        # create diffusion model
        assert obs_as_global_cond, 'assert obs_as_global_cond'

        input_dim = action_dim
        # global_cond_dim = obs_feature_dim * n_obs_steps
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
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
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, batch) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        instructions: language instruction
        img_instructions: image instruction
        """
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'], ignore_keys=self.rgb_keys)
        B = nobs['agent_pos'].shape[0]
        T = self.action_horizon
        Da = self.action_dim
        Do = self.obs_feature_dim

        instructions = batch['instruction'] if 'instruction' in batch else None
        # 目前只实现一条sample一个instruction的版本
        if instructions and isinstance(instructions[0], list):
            instructions = [instruction[0] for instruction in instructions]
        global_cond = None
        # B,T,C,H,W
        this_nobs = dict_apply(nobs, lambda x: x, ignore_keys=['agent_pos']) # 不处理agent_pos，在外面处理
        # build input
        device = self.device
        dtype = self.dtype
        # condition through global feature
        if self.mask_type == 'mask_only' and self.mask_keys and len(self.mask_keys) > 0:
            for mask_key in self.mask_keys:
                this_nobs[mask_key] = batch[mask_key]
        nobs_features = self.obs_encoder(this_nobs, instructions)
        # agent_pos
        agent_pos = nobs['agent_pos'].reshape(B, -1) # B,T*Da
        nobs_features = torch.concat([nobs_features, agent_pos], axis=-1) # 
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

    def forward(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # T = self.action_horizon
        nobs = self.normalizer.normalize(batch['obs'], ignore_keys=self.rgb_keys)
        B = nobs['agent_pos'].shape[0]
        nactions = self.normalizer['action'].normalize(batch['action'])
        instructions = batch['instruction'] if 'instruction' in batch else None
        # 目前只实现一条sample一个instruction的版本
        if instructions and isinstance(instructions[0], list):
            instructions = [instruction[0] for instruction in instructions]
        global_cond = None
        # B,T,C,H,W
        this_nobs = dict_apply(nobs, lambda x: x, ignore_keys=['agent_pos']) # 不处理agent_pos，在外面处理
        # condition through global feature
        if self.mask_type == 'mask_only' and self.mask_keys and len(self.mask_keys) > 0:
            for mask_key in self.mask_keys:
                this_nobs[mask_key] = batch[mask_key]
        nobs_features = self.obs_encoder(this_nobs, instructions) # B,Do
        # agent_pos
        agent_pos = nobs['agent_pos'].reshape(B, -1) # B,T*Da
        nobs_features = torch.concat([nobs_features, agent_pos], axis=-1) # 

        trajectory = nactions
        global_cond = nobs_features.reshape(B, -1)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        return pred, target

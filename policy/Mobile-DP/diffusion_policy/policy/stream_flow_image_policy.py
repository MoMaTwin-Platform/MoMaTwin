from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d_streamflow import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from x2robot_dataset.common.data_utils import decode_text
import random
import torch.distributions as distributions

class StreamFlowPolicy(BaseImagePolicy):
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

        # these two hyperparameters are set according to official notebook: https://siddancha.github.io/streaming-flow-policy/notebooks/stabilizing-sfp.html
        self.k = 2.5
        self.sigma_0 = 0.05

        self.action_chunk_size_when_training = self.action_horizon
        self.action_chunk_size_when_infering = self.action_horizon

    def register_visual_instruct(self, visual_instructor):
        self.visual_instructor = visual_instructor

    def sample_t(self, chunk_size):
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size必须是正整数")
        
        # 在[0, chunk_size)区间均匀采样一个整数t
        t = random.randrange(chunk_size)
        
        # 计算归一化值（总共chunk_size个action，得到的时间区间个数是chunk_size-1）
        t_norm = t / (chunk_size - 1)
        
        return t, t_norm

    def sample_action_from_gaussian_tube(self, t: int, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        根据给定的时间t、动作序列、初始方差和衰减率,
        构造一个高斯分布并从中采样一个动作。

        Args:
            t (int): 时间步,范围从1到chunk_length (1 <= t <= chunk_length)。
            action_chunk (torch.Tensor): 动作序列,形状为 (batch_size, chunk_length, action_dim)。
            sigma_0 (float): 初始标准差的系数。
            k (float): 指数衰减率。

        Returns:
            torch.Tensor: 从构造的高斯分布中采样得到的动作,形状为 (batch_size, 1, action_dim)。
        """
        
        # 1. 根据时刻t取出t时刻的action作为高斯分布的均值
        mean_action = action_chunk[:, t, :]

        # 2. 计算高斯分布的方差: σ₀²e⁻²ᵏᵗ
        # 公式中的t通常指归一化后的时间 t ∈ [0, 1]。因此,我们需要将输入的整数 t ( 0 <= t < chunk_length) 归一化到 [0, 1) 范围。
        chunk_length = action_chunk.shape[1]
        
        if chunk_length > 1:
            # 归一化时间 t: 从0到chunk_length-1映射到0到1
            t_normalized = t / (chunk_length - 1)
        else:
            # 如果chunk_length为1, t必然为0, 此时 t_normalized 设为1
            t_normalized = 1.0

        dtype = action_chunk.dtype
        device = action_chunk.device

        sigma_0_tensor = torch.tensor(self.sigma_0, dtype=dtype, device=device)
        k_tensor = torch.tensor(self.k, dtype=dtype, device=device)
        t_normalized_tensor = torch.tensor(t_normalized, dtype=dtype, device=device)

        # 计算方差
        variance = (sigma_0_tensor ** 2) * torch.exp(-2 * k_tensor * t_normalized_tensor)

        # 从方差获取标准差 (PyTorch的Normal分布期望的是标准差,而不是方差)
        std_dev = torch.sqrt(variance)

        # 3. 构造高斯分布
        dist = distributions.Normal(loc=mean_action, scale=std_dev)

        # 4. 从分布中采样一个动作
        sampled_action = dist.sample()  # (batch_size, action_dim)

        # 5. 返回所采样的动作,并确保形状为 (batch_size, 1, action_dim)
        return sampled_action.unsqueeze(1)


    def get_target_velocity(self, sampled_action: torch.Tensor, sampled_t: int, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        根据给定的公式 v_ξ(a, t) = ξ̇(t) - k(a - ξ(t)) 构造目标瞬时速度真值。

        Args:
            sampled_action (torch.Tensor): 从高斯分布中采样的动作，形状为 (batch_size, 1, action_dim)。
                                        对应公式中的 'a'。
            sampled_t (int): 从 [0, chunk_size-1] 区间中采样的一个时间步。
                            对应公式中的 't'。
            action_chunk (torch.Tensor): 原始动作序列 (轨迹 ξ)，形状为 (batch_size, chunk_length, action_dim)。
            k (float): 稳定项系数。

        Returns:
            torch.Tensor: 构造出的目标瞬时速度真值，形状为 (batch_size, 1, action_dim)。
                        对应公式中的 'v_ξ(a, t)'。
        """
        
        batch_size, chunk_length, action_dim = action_chunk.shape
        dtype = action_chunk.dtype
        device = action_chunk.device

        # --- 1. 计算 ξ(t) ---
        # ξ(t) 是在原始轨迹 action_chunk 中，时间步 sampled_t 的动作。
        xi_t = action_chunk[:, sampled_t, :] # Shape: (batch_size, action_dim)
        
        # 为了与 sampled_action (batch_size, 1, action_dim) 进行操作，将 xi_t 也扩展一个维度。
        xi_t_expanded = xi_t.unsqueeze(1) # Shape: (batch_size, 1, action_dim)

        # --- 2. 计算轨迹速度 (Trajectory velocity): ξ̇(t) ---
        trajectory_velocity_term = torch.zeros_like(xi_t_expanded) # 初始化为零，应对 chunk_length <= 1 的情况

        if chunk_length <= 1:
            # 如果轨迹长度为1或更短，无法计算有效的速度差。
            # 此时速度被认为是0。
            # trajectory_velocity_term 保持为全零张量
            pass 
        else:
            # 归一化时间步长 dt = 1 / (chunk_length - 1)
            # 注意：这里如果 chunk_length 是1，(chunk_length - 1) 就是0，会报错。
            # 但我们已经在 chunk_length <= 1 的条件中处理了。
            dt = 1.0 / (chunk_length - 1)
            dt_tensor = torch.tensor(dt, dtype=dtype, device=device) # 将dt转换为tensor，保持数据类型和设备一致

            if sampled_t == 0:
                # 初始动作的速度被设置为第二个动作的速度。
                # ξ̇(0) = (action_chunk[1] - action_chunk[0]) / dt
                # 这里安全是因为 chunk_length > 1 保证了 action_chunk[:, 1, :] 可用
                delta_action = action_chunk[:, 1, :] - action_chunk[:, 0, :]
                trajectory_velocity_term = (delta_action / dt_tensor).unsqueeze(1)
            else:
                # 对于 sampled_t > 0 的情况，使用 (当前 - 上一个) / dt
                # ξ̇(t) = (action_chunk[t] - action_chunk[t-1]) / dt
                delta_action = action_chunk[:, sampled_t, :] - action_chunk[:, sampled_t - 1, :]
                trajectory_velocity_term = (delta_action / dt_tensor).unsqueeze(1)
        
        # --- 3. 计算稳定项 (Stabilization term): -k(a - ξ(t)) 这里的 'a' 就是 sampled_action
        k_tensor = torch.tensor(self.k, dtype=dtype, device=device)
        stabilization_term = -k_tensor * (sampled_action - xi_t_expanded)
        # 稳定项的形状应为 (batch_size, 1, action_dim)

        # --- 4. 组合计算最终的目标速度 ---
        # v_ξ(a,t) = ξ̇(t) - k(a - ξ(t))
        target_velocity = trajectory_velocity_term + stabilization_term

        return target_velocity

    def get_pred_and_target(self, action_chunk: torch.Tensor, global_cond: torch.Tensor):
        chunk_size = action_chunk.shape[1]

        # 1. 采样时间步 t
        t, t_norm = self.sample_t(chunk_size)

        # 2. 采样动作 a (从训练集的动作构造的高斯管道中采样)
        sampled_action = self.sample_action_from_gaussian_tube(t, action_chunk)

        # 3. 计算目标速度 v_ξ(a, t)
        target_velocity = self.get_target_velocity(sampled_action, t, action_chunk)

        # 4. 网络预测速度
        timesteps = torch.full((action_chunk.shape[0],), t, device=global_cond.device)
        pred_velocity = self.model(
            sampled_action,             # (batch_size, 1, action_dim)
            timesteps,                  # (batch_size)
            global_cond=global_cond     # (batch_size, global_cond_dim)
        )

        return pred_velocity, target_velocity

    def predict_action_chunk(self, cur_action: torch.Tensor, global_cond: torch.Tensor, delta_t=None):
        # 如果不指定delta_t，则默认为训练时的时间步长
        if delta_t is None:
            delta_t = 1.0 / (self.action_chunk_size_when_training - 1)

        cur_t = 0
        time_horizon_norm = self.action_chunk_size_when_infering/self.action_chunk_size_when_training
        pred_action_ls = []
        while cur_t <= time_horizon_norm:
            pred_velocity = self.model(cur_action, cur_t, global_cond=global_cond)
            pred_action = cur_action + pred_velocity * delta_t
            cur_t += delta_t
            pred_action_ls.append(pred_action)
            cur_action = pred_action
        pred_action_chunk = torch.cat(pred_action_ls, dim=1)

        return pred_action_chunk

    def predict_action_stream(self, cur_action: torch.Tensor, global_cond: torch.Tensor, cur_t = 0, delta_t=None):
        # 如果不指定delta_t，则默认为训练时的时间步长
        if delta_t is None:
            delta_t = 1.0 / (self.action_chunk_size_when_training - 1)

        pred_velocity = self.model(cur_action, cur_t, global_cond=global_cond)
        pred_action = cur_action + pred_velocity * delta_t
        cur_t += delta_t

        return pred_action, cur_t

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

        # 使用obs['agent_pos']作为cur_action
        cur_action = nobs['agent_pos']
        nsample = self.predict_action_chunk(cur_action, global_cond)
        
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

        pred_velocity, target_velocity = self.get_pred_and_target(trajectory, global_cond)

        return pred_velocity, target_velocity

    def forward(self, batch, predict=False):
        if predict:
            return self.predict_action(batch)
        else:
            return self.train_step_forward(batch)


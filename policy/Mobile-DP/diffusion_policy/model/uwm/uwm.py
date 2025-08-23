from diffusion_policy.model.common.normalizer import LinearNormalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange
from transformers import CLIPTokenizer


from diffusion_policy.model.common.adaln_attention import AdaLNAttentionBlock, AdaLNFinalLayer
from diffusion_policy.model.common.utils import SinusoidalPosEmb, init_weights
from diffusion_policy.model.uwm.obs_encoder import UWMObservationEncoder
from x2robot_dataset.common.data_utils import decode_text


class MultiViewVideoPatchifier(nn.Module):
    def __init__(
        self,
        num_views: int,
        input_shape: tuple[int, ...] = (8, 224, 224),
        patch_shape: tuple[int, ...] = (2, 8, 8),
        num_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_views = num_views
        iT, iH, iW = input_shape
        pT, pH, pW = patch_shape
        self.T, self.H, self.W = iT // pT, iH // pH, iW // pW
        self.pT, self.pH, self.pW = pT, pH, pW

        self.patch_encoder = nn.Conv3d(
            in_channels=num_chans,
            out_channels=embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
        )
        self.patch_decoder = nn.Linear(embed_dim, num_chans * pT * pH * pW)

    def forward(self, imgs):
        return self.patchify(imgs)

    def patchify(self, imgs):
        imgs = rearrange(imgs, "b v c t h w -> (b v) c t h w")
        # print(f'imgs: {imgs.shape}', flush=True)
        feats = self.patch_encoder(imgs)
        # print(f'feats1: {feats.shape}', flush=True)
        feats = rearrange(feats, "(b v) c t h w -> b (v t h w) c", v=self.num_views)
        # print(f'feats2: {feats.shape}', flush=True)
        return feats

    def unpatchify(self, feats):
        imgs = self.patch_decoder(feats)
        imgs = rearrange(
            imgs,
            "b (v t h w) (c pt ph pw) -> b v c (t pt) (h ph) (w pw)",
            v=self.num_views,
            t=self.T,
            h=self.H,
            w=self.W,
            pt=self.pT,
            ph=self.pH,
            pw=self.pW,
        )
        return imgs

    @property
    def num_patches(self):
        return self.num_views * self.T * self.H * self.W


class DualTimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0):
        super().__init__()
        self.sinusoidal_pos_emb = SinusoidalPosEmb(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t1, t2):
        temb1 = self.sinusoidal_pos_emb(t1)
        temb2 = self.sinusoidal_pos_emb(t2)
        temb = torch.cat([temb1, temb2], dim=-1)
        return self.proj(temb)


class DualNoisePredictionNet(nn.Module):
    def __init__(
        self,
        global_cond_dim: int,
        image_shape: tuple[int, ...],
        patch_shape: tuple[int, ...],
        num_chans: int,
        num_views: int,
        action_len: int,
        action_dim: int,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        num_registers: int = 8,
    ):
        super().__init__()
        # Observation encoder and decoder
        self.obs_patchifier = MultiViewVideoPatchifier(
            num_views=num_views,
            input_shape=image_shape,
            patch_shape=patch_shape,
            num_chans=num_chans,
            embed_dim=embed_dim,
        )
        obs_len = self.obs_patchifier.num_patches

        # Action encoder and decoder
        hidden_dim = int(max(action_dim, embed_dim) * mlp_ratio)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Timestep embedding
        self.timestep_embedding = DualTimestepEncoder(timestep_embed_dim)

        # Registers
        self.registers = nn.Parameter(
            torch.empty(1, num_registers, embed_dim).normal_(std=0.02)
        )

        # Positional embedding
        total_len = action_len + obs_len + num_registers
        self.pos_embed = nn.Parameter(
            torch.empty(1, total_len, embed_dim).normal_(std=0.02)
        )
        # self.register_buffer('pos_embed', torch.zeros(1, 1, embed_dim), persistent=False)  # 占位符
        # DiT blocks
        cond_dim = global_cond_dim + timestep_embed_dim
        self.blocks = nn.ModuleList(
            [
                AdaLNAttentionBlock(
                    dim=embed_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                )
                for _ in range(depth)
            ]
        )
        self.head = AdaLNFinalLayer(dim=embed_dim, cond_dim=cond_dim)
        self.action_inds = (0, action_len)
        self.next_obs_inds = (action_len, action_len + obs_len)

        # AdaLN-specific weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Base initialization
        self.apply(init_weights)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.obs_patchifier.patch_encoder.weight.data
        nn.init.normal_(w.view([w.shape[0], -1]), mean=0.0, std=0.02)
        nn.init.constant_(self.obs_patchifier.patch_encoder.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def forward(self, global_cond, action, action_t, next_obs, next_obs_t):
        # Encode inputs
        action_embed = self.action_encoder(action)
        # print(f'next_obs: {next_obs.shape}', flush=True)
        next_obs_embed = self.obs_patchifier(next_obs)

        # Expand and encode timesteps
        if len(action_t.shape) == 0:
            action_t = action_t.expand(action.shape[0]).to(
                dtype=torch.long, device=action.device
            )
        if len(next_obs_t.shape) == 0:
            next_obs_t = next_obs_t.expand(next_obs.shape[0]).to(
                dtype=torch.long, device=next_obs.device
            )
        temb = self.timestep_embedding(action_t, next_obs_t)
        # print(f'action_t: {action_t.shape}, next_obs_t: {next_obs_t.shape}, temb: {temb.shape}', flush=True)

        # Forward through model
        registers = self.registers.expand(next_obs.shape[0], -1, -1)
        # print(f'action_embed: {action_embed.shape}, next_obs_embed: {next_obs_embed.shape}, registers: {registers.shape}', flush=True)
        x = torch.cat((action_embed, next_obs_embed, registers), dim=1)
        # print(f'x: {x.shape}, self.pos_embed: {self.pos_embed.shape}', flush=True)

        # 动态生成位置嵌入
        # if self.pos_embed.shape[1] != x.shape[1]:
        #     pos_embed = torch.empty(1, x.shape[1], x.shape[2], device=x.device).normal_(std=0.02)
        #     self.register_buffer('pos_embed', pos_embed, persistent=False)
        
        x = x + self.pos_embed

        cond = torch.cat((global_cond, temb), dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        x = self.head(x, cond)

        # Extract action and next observation noise predictions
        action_noise_pred = x[:, self.action_inds[0] : self.action_inds[1]]
        next_obs_noise_pred = x[:, self.next_obs_inds[0] : self.next_obs_inds[1]]

        # Decode outputs
        action_noise_pred = self.action_decoder(action_noise_pred)
        next_obs_noise_pred = self.obs_patchifier.unpatchify(next_obs_noise_pred)
        return action_noise_pred, next_obs_noise_pred


class UnifiedWorldModel(nn.Module):
    def __init__(
        self,
        action_len: int,
        action_dim: int,
        obs_encoder: UWMObservationEncoder,
        embed_dim: int = 768,
        timestep_embed_dim: int = 512,
        latent_patch_shape: tuple[int, ...] = (2, 4, 4),
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        num_registers: int = 8,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
    ):
        """
        Assumes rgb input: (B, T, H, W, C) uint8 image
        Assumes low_dim input: (B, T, D)
        """
        super().__init__()
        self.rgb_keys = ['face_view', 'left_wrist_view', 'right_wrist_view']
        self.action_len = action_len
        self.action_dim = action_dim
        self.action_shape = (action_len, action_dim)

        # Image augmentation
        self.obs_encoder = obs_encoder
        self.latent_img_shape = self.obs_encoder.latent_img_shape()

        # Diffusion noise prediction network
        global_cond_dim = self.obs_encoder.feat_dim()
        image_shape = self.latent_img_shape[2:]
        num_views, num_chans = self.latent_img_shape[:2]
        self.noise_pred_net = DualNoisePredictionNet(
            global_cond_dim=global_cond_dim,
            image_shape=image_shape,
            patch_shape=latent_patch_shape,
            num_chans=num_chans,
            num_views=num_views,
            action_len=action_len,
            action_dim=action_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            num_registers=num_registers,
        )

        # Diffusion scheduler
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )
        # normalizer
        self.normalizer = LinearNormalizer()
        
        # Initialize CLIP tokenizer for language processing (only if needed)
        if self.obs_encoder.use_language:
            self.tokenizer = CLIPTokenizer.from_pretrained("/x2robot/Models/clip-vit-base-patch32")
        else:
            self.tokenizer = None


    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _add_language_tokens_if_needed(self, obs_dict, instruction=""):
        """为obs_dict添加语言tokens（仅当use_language=True时），用于sample方法"""
        if self.obs_encoder.use_language and self.tokenizer is not None and "input_ids" not in obs_dict:
            # 推断batch_size
            first_key = next(iter(obs_dict.keys()))
            batch_size = obs_dict[first_key].shape[0]
            device = obs_dict[first_key].device
            
            # 为默认instruction创建token
            instructions = [instruction] * batch_size
            tokenized = self.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            
            obs_dict["input_ids"] = tokenized["input_ids"].to(device)
            obs_dict["attention_mask"] = tokenized["attention_mask"].to(device)
        
        return obs_dict

    def process_batch(self, batch):
        curr_obs = {k: v for k, v in batch["obs"].items()}
        next_obs = {k: v for k, v in batch["next_obs"].items()}
        # 先支持主摄像头的视频生成
        actions = batch["action"]
    
        # 获取语言指令
        instructions = batch['instruction'] if 'instruction' in batch else None
        if isinstance(instructions, torch.Tensor):
            instructions = [decode_text(instruction) for instruction in instructions]
        if instructions and isinstance(instructions[0], list):
            instructions = [instruction[0] for instruction in instructions]
            
        
        # Add language tokens - 只有当obs_encoder启用了use_language时才处理
        if self.obs_encoder.use_language:
            if "input_ids" in batch and "attention_mask" in batch:
                curr_obs["input_ids"] = batch["input_ids"]
                curr_obs["attention_mask"] = batch["attention_mask"]
            elif instructions is not None and len(instructions) > 0:
                # 使用tokenizer处理instruction文本
                # 确保所有instruction都是字符串
                instructions = [str(inst) if inst is not None else "" for inst in instructions]
                
                # 使用tokenizer进行编码
                tokenized = self.tokenizer(
                    instructions,
                    padding=True,
                    truncation=True,
                    max_length=77,  # CLIP的标准长度
                    return_tensors="pt"
                )
                
                # 移动到正确的设备
                device = actions.device
                curr_obs["input_ids"] = tokenized["input_ids"].to(device)
                curr_obs["attention_mask"] = tokenized["attention_mask"].to(device)
        return curr_obs, next_obs, actions
    
    # def forward(self, obs_dict, next_obs_dict, action, action_mask=None):
    def forward(self, batch, predict=None):
        if predict == 'action':
            return self.sample(batch['obs'])
        elif predict == 'joint':
            return self.sample_joint(batch['obs'])
        elif predict == 'next_obs':
            return self.sample_marginal_next_obs(batch['obs'])
        else:
            return self.train_forward(batch)

    def train_forward(self, batch):
        obs_dict, next_obs_dict, action = self.process_batch(batch)
        action_mask = None
        if "action_mask" in batch:
            action_mask = batch["action_mask"]
        
        # 获取训练目标
        train_target = batch.get('train_target', 'joint')
        
        batch_size, device = action.shape[0], action.device

        # normalize low dim obs and predicted action
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        action = self.normalizer['action'].normalize(action)

        # Encode observations, 他这里要求curr和next obs是一样的长度, 真的有必要吗？
        obs, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Sample diffusion timestep for action
        action_noise = torch.randn_like(action)
        action_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        if action_mask is not None:
            action_t[~action_mask] = self.num_train_steps - 1
        noisy_action = self.noise_scheduler.add_noise(action, action_noise, action_t)

        # Sample diffusion timestep for next observation
        next_obs_noise = torch.randn_like(next_obs)
        next_obs_t = torch.randint(
            low=0, high=self.num_train_steps, size=(batch_size,), device=device
        ).long()
        noisy_next_obs = self.noise_scheduler.add_noise(
            next_obs, next_obs_noise, next_obs_t
        )

        # Diffusion loss
        action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
            obs, noisy_action, action_t, noisy_next_obs, next_obs_t
        )
        
        # 根据训练目标计算损失
        action_loss = torch.tensor(0.0, device=device, requires_grad=True)
        dynamics_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if train_target in ['action', 'joint']:
            # 计算动作损失
            if action_mask is not None:
                if action_mask.any():
                    action_loss = F.mse_loss(action_noise_pred[action_mask], action_noise[action_mask])
            else:
                action_loss = F.mse_loss(action_noise_pred, action_noise)
        
        if train_target in ['next_obs', 'joint']:
            # 计算视频生成损失
            dynamics_loss = F.mse_loss(next_obs_noise_pred, next_obs_noise)
        
        # 组合损失
        if train_target == 'action':
            loss = action_loss
        elif train_target == 'next_obs':
            loss = dynamics_loss
        else:  # joint
            loss = action_loss + dynamics_loss

        # Logging
        info = {
            "loss": loss.item(),
            "action_loss": action_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "train_target": train_target
        }
        return loss, info

    @torch.no_grad()
    def sample(self, obs_dict):
        return self.sample_marginal_action(obs_dict)

    @torch.no_grad()
    def sample_forward_dynamics(self, obs_dict, action):
        # normalize low dim obs
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        
        # Add language tokens if needed
        obs_dict = self._add_language_tokens_if_needed(obs_dict)

        # Encode observations
        obs = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize next observation sample
        next_obs_sample = torch.randn(
            (obs.shape[0],) + self.latent_img_shape, device=obs.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[-1]
        for next_obs_t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs, action, action_t, next_obs_sample, next_obs_t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, next_obs_t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_inverse_dynamics(self, obs_dict, next_obs_dict):
        # normalize low dim obs
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        
        # Add language tokens if needed
        obs_dict = self._add_language_tokens_if_needed(obs_dict)

        # Encode observations
        obs_feat, next_obs = self.obs_encoder.encode_curr_and_next_obs(
            obs_dict, next_obs_dict
        )

        # Initialize action sample
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[-1]
        for action_t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, action_t, action_sample
            ).prev_sample
        # unnormalize action
        action_sample = self.normalizer['action'].unnormalize(action_sample)
        return action_sample

    @torch.no_grad()
    def sample_marginal_next_obs(self, obs_dict):
        # normalize low dim obs
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        
        # Add language tokens if needed
        obs_dict = self._add_language_tokens_if_needed(obs_dict)

        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        action_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            _, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, action_t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
        return next_obs_sample

    @torch.no_grad()
    def sample_marginal_action(self, obs_dict):
        # normalize low dim obs
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        
        # Add language tokens if needed
        obs_dict = self._add_language_tokens_if_needed(obs_dict)

        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        next_obs_t = self.noise_scheduler.timesteps[0]
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, _ = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, next_obs_t
            )
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample

        # unnormalize action
        action_sample = self.normalizer['action'].unnormalize(action_sample)
        return action_sample

    @torch.no_grad()
    def sample_joint(self, obs_dict):
        # normalize low dim obs
        obs_dict = self.normalizer.normalize(obs_dict, ignore_keys=self.rgb_keys)
        
        # Add language tokens if needed
        obs_dict = self._add_language_tokens_if_needed(obs_dict)

        obs_feat = self.obs_encoder.encode_curr_obs(obs_dict)

        # Initialize action and next_obs
        action_sample = torch.randn(
            (obs_feat.shape[0],) + self.action_shape, device=obs_feat.device
        )
        next_obs_sample = torch.randn(
            (obs_feat.shape[0],) + self.latent_img_shape, device=obs_feat.device
        )

        # Sampling steps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            action_noise_pred, next_obs_noise_pred = self.noise_pred_net(
                obs_feat, action_sample, t, next_obs_sample, t
            )
            next_obs_sample = self.noise_scheduler.step(
                next_obs_noise_pred, t, next_obs_sample
            ).prev_sample
            action_sample = self.noise_scheduler.step(
                action_noise_pred, t, action_sample
            ).prev_sample
        # unnormalize action
        action_sample = self.normalizer['action'].unnormalize(action_sample)
        return next_obs_sample, action_sample

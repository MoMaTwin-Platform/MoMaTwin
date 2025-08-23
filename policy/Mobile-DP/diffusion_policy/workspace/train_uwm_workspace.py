if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import gc
import os
import copy
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, TactileLinearNormalizer
from diffusion_policy.model.uwm import UnifiedWorldModel
from diffusion_policy.model.uwm.obs_encoder import VideoTransform
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import torch.distributed
from torchvision.utils import make_grid
import copy
import random
import wandb
import tqdm
import time
import accelerate
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusers.utils.torch_utils import is_compiled_module
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from einops import rearrange, reduce
from x2robot_dataset.dataloader import DynamicDataLoader
from x2robot_dataset.common.collate_fn import collate_wrapper
from x2robot_dataset.common.data_preprocessing import _ACTION_KEY_WITH_HAND_MAPPING, _CAM_MAPPING, _CAM_FILE_MAPPING, _CAM_BINOCULAR_FILE_MAPPING, _ACTION_KEY_EE_MAPPING
from x2robot_dataset.common.data_utils import relative_to_actions
from x2robot_dataset.lazy_dataset import (
    IterChunkDataset,
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dataloader import DynamicDataLoader
OmegaConf.register_new_resolver("eval", eval, replace=True)

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

def compute_loss(pred, target):
    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss.type(loss.dtype)
    loss = reduce(loss, 'b ... -> b (...)', 'mean')
    loss = loss.mean()
    return loss
    
def simple_wandb_log(wandb_run, step_log, step, max_retries=3):
    wandb_run.log(step_log, step=step)
    return True

def safe_wandb_log(wandb_run, step_log, step, max_retries=3, log_interval=100):
    """只在特定步数间隔记录日志"""
    if step % log_interval != 0:
        return True
        
    for attempt in range(max_retries):
        try:
            wandb_run.log(step_log, step=step)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to log to wandb after {max_retries} attempts: {e}")
                return False
            time.sleep(1 * (attempt + 1))

def print_rank0(accelerator, msg):
    if accelerator.is_main_process:
        print(msg, flush=True)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

class TrainUWMWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # configure 
        self.rgb_keys = list()
        self.lowdim_keys = list()
        self.tactile_keys = list()
        obs_shape_meta = cfg.task.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                self.rgb_keys.append(key)
            elif type == 'low_dim':
                self.lowdim_keys.append(key)
            elif type == 'tactile':
                self.tactile_keys.append(key)
        print(self.rgb_keys)
        print(self.lowdim_keys)
        print(self.tactile_keys)
        self.use_tactile = len(self.tactile_keys) > 0
        self.action_dim = cfg.task.shape_meta['action'].shape[0]
        self.low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
        self.img_obs_horizon = cfg.task.img_obs_horizon
        self.action_horizon = cfg.task.action_horizon

        self.minmax_range_robot = default(cfg, 'task.minmax_range_robot', 'arx') # 是否是arx,leju,leju_v2
        # configure model
        self.model: UnifiedWorldModel = hydra.utils.instantiate(cfg.policy)

        param_groups = [
            {'params': self.model.parameters()},
        ]
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )
        save_optimizer = default(cfg, 'checkpoint.save_optimizer', False)
        if not save_optimizer:
            self.exclude_keys = ['optimizer']

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.fsdp = False

        # 添加训练目标配置
        self.train_target = default(cfg, 'training.train_target', 'joint')  # 支持 'action', 'joint', 'next_obs'
        
        # 验证配置
        valid_targets = ['action', 'joint', 'next_obs']
        if self.train_target not in valid_targets:
            raise ValueError(f"train_target must be one of {valid_targets}, got {self.train_target}")
        
        print(f"Training target: {self.train_target}")

    def _set_action_mask_for_target(self, batch, train_target):
        """根据训练目标设置action_mask"""
        device = batch['action'].device
        batch_size = batch['action'].shape[0]
        
        if train_target == 'next_obs':
            # 只训练视频生成：action_mask全为False
            batch['action_mask'] = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif train_target == 'action':
            # 只训练动作预测：action_mask全为True，但需要在模型中停止next_obs的梯度
            batch['action_mask'] = torch.ones(batch_size, dtype=torch.bool, device=device)
            batch['train_target'] = 'action'  # 传递给模型
        elif train_target == 'joint':
            # 联合训练：action_mask全为True
            batch['action_mask'] = torch.ones(batch_size, dtype=torch.bool, device=device)
            batch['train_target'] = 'joint'  # 传递给模型
        
        return batch

    def train_loop(self, train_num, rank, accelerator, train_dataloader, lr_scheduler, wandb_run, json_logger, cfg, step_log, local_epoch_idx, gradient_accumulation_steps):
        # ========= train for this epoch ==========
        step_loss = 0.0
        tepoch = tqdm.tqdm(total=train_num, desc=f"rank {rank} - Training epoch {self.epoch} ({self.train_target})", leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process)
        self.model.train()
        
        torch.cuda.nvtx.range_push("batch 0")
        torch.cuda.nvtx.range_push("data load")
        for batch_idx, batch in enumerate(train_dataloader):
            torch.cuda.nvtx.range_pop()
            if batch_idx >= train_num-1:
                print(f"rank {rank} batch_idx {batch_idx} >= train_num {train_num} -1, break")
                break

            # 根据训练目标设置action_mask
            batch = self._set_action_mask_for_target(batch, self.train_target)

            with accelerator.accumulate(self.model):
                # compute loss
                with torch.cuda.nvtx.range(f"forward compute"):
                    loss, info = self.model(batch)
                with torch.cuda.nvtx.range(f"backward compute"):
                    accelerator.backward(loss)
                with torch.cuda.nvtx.range(f"optimizer step"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()

            step_loss = loss.detach().item()
            step_log.update({
                'train_loss': step_loss,
                'global_step': self.global_step,
                'epoch': self.epoch,
                'lr': lr_scheduler.get_last_lr()[0],
                'train_target': self.train_target  # 记录训练目标
            })
            for k, v in info.items():
                step_log.update({
                    f'train/{k}': v
                })
            
            if accelerator.is_main_process and self.global_step % gradient_accumulation_steps == 0:
                # logging
                tepoch.set_postfix(loss=step_loss, refresh=False)
                tepoch.update(1)
                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    if not safe_wandb_log(wandb_run, step_log, step=self.global_step, log_interval=20):
                        print(f"Failed to log to wandb", flush=True)
                    json_logger.log(step_log)
            self.global_step += 1

            torch.cuda.nvtx.range_pop()
            
            if batch_idx <= train_num - 2:
                torch.cuda.nvtx.range_push(f"batch {batch_idx+1}")
                torch.cuda.nvtx.range_push("data load")

    def validation_loop(self, val_num, rank, accelerator, val_dataloader, cfg, step_log):
        self.model.eval()
        with torch.no_grad():
            val_losses = list()
            val_action_losses = list()
            val_dynamics_losses = list()
            
            with tqdm.tqdm(total=val_num, desc=f"rank {rank} - Validation epoch {self.epoch} ({self.train_target})", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as vepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= val_num-1:
                        break
                    
                    # 为验证设置相同的训练目标
                    batch = self._set_action_mask_for_target(batch, self.train_target)
                    loss, info = self.model(batch)
                    
                    # 修改: 先在本地累积loss，最后再同步一次
                    val_losses.append(loss.detach())
                    val_action_losses.append(torch.tensor(info['action_loss'], device=loss.device))
                    val_dynamics_losses.append(torch.tensor(info['dynamics_loss'], device=loss.device))
                    
                    if accelerator.is_local_main_process:
                        vepoch.update(1)
                        
            # 修改: 在所有batch处理完后，只进行一次gather操作
            if len(val_losses) > 0:
                val_loss = torch.mean(torch.stack(val_losses))
                val_action_loss = torch.mean(torch.stack(val_action_losses))
                val_dynamics_loss = torch.mean(torch.stack(val_dynamics_losses))
                
                gathered_loss = accelerator.gather(val_loss)
                gathered_action_loss = accelerator.gather(val_action_loss)
                gathered_dynamics_loss = accelerator.gather(val_dynamics_loss)
                
                if accelerator.is_main_process:
                    step_log['val_loss'] = gathered_loss.mean().item()
                    
                    # 根据训练目标记录相应的损失
                    if self.train_target in ['action', 'joint']:
                        step_log['val_action_loss'] = gathered_action_loss.mean().item()
                    if self.train_target in ['next_obs', 'joint']:
                        step_log['val_dynamics_loss'] = gathered_dynamics_loss.mean().item()
                
        accelerator.wait_for_everyone()

    def predict_loop(self, val_num, rank, accelerator, val_dataloader, cfg, step_log, use_quaternion=False, parse_head_action=False, parse_head_action_v2=False, relative_action=False):
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_actions = []
            # 为每个视角单独收集预测结果
            view_obs_preds = {view: [] for view in self.rgb_keys}
            view_obs_gts = {view: [] for view in self.rgb_keys}
            
            # 根据训练目标决定是否生成图像
            should_generate_images = self.train_target in ['next_obs', 'joint']
            should_predict_actions = self.train_target in ['action', 'joint']
            
            # 设置最大图像生成批次数，只在需要生成图像时使用
            max_image_generations = default(cfg, 'training.max_image_generations', 0) if should_generate_images else 0
            current_image_generations = 0
            
            with tqdm.tqdm(total=val_num, desc=f"rank {rank} - predict epoch {self.epoch} ({self.train_target})", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                    disable=not accelerator.is_local_main_process) as pepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= val_num-1:
                        break
                    
                    gt_action = batch['action']
                    next_obs_sample = None
                    pred_action = None
                    
                    # 根据训练目标进行预测
                    if self.train_target == 'action':
                        # 只预测动作
                        result = self.model(batch, predict='action')
                        pred_action = result.to(self.model.device)
                        
                    elif self.train_target == 'next_obs':
                        # 只预测视频，但需要控制生成数量
                        if current_image_generations < max_image_generations:
                            current_image_generations += 1
                            next_obs_sample = self.model(batch, predict='next_obs')
                            # 对于next_obs模式，我们不预测action，所以pred_action保持None
                            
                    elif self.train_target == 'joint':
                        # 联合预测，保持原有逻辑
                        if current_image_generations < max_image_generations:
                            current_image_generations += 1
                            gt_next_obs = batch.get('next_obs', None)
                            next_obs_sample, result = self.model(batch, predict='joint')
                            pred_action = result.to(self.model.device)
                        else:
                            # 只预测动作
                            result = self.model(batch, predict='action')
                            pred_action = result.to(self.model.device)
                    
                    # 处理动作预测评估（只在需要时）
                    if should_predict_actions and pred_action is not None:
                        # 处理相对动作转换
                        if relative_action:
                            agent_pos = batch['obs']['agent_pos'].cpu().numpy()
                            pred_action = pred_action.cpu().numpy() 
                            abs_action = []
                            for apos, paction in zip(agent_pos, pred_action):
                                abs_action.append(relative_to_actions(paction, apos[0]))
                            pred_action = torch.tensor(abs_action, device=self.model.device)
                        
                        gt_action = gt_action.to(self.model.device)
                        
                        # 确保所有进程的数据大小一致
                        pred_action, gt_action = accelerator.pad_across_processes(
                            [pred_action, gt_action],
                            dim=0
                        )
                        
                        # 收集所有GPU的预测结果
                        gathered_preds = accelerator.gather(pred_action)
                        gathered_actions = accelerator.gather(gt_action)
                        
                        if accelerator.is_main_process:
                            all_preds.append(gathered_preds)
                            all_actions.append(gathered_actions)
                    
                    # 处理图像预测（只在需要时）
                    if should_generate_images and next_obs_sample is not None:
                        gt_next_obs = batch.get('next_obs', None)
                        if gt_next_obs is not None:
                            # 分别处理每个视角
                            for view_idx, view_key in enumerate(self.rgb_keys):
                                # 提取对应视角的预测和真实图像
                                view_pred = next_obs_sample[:, view_idx:view_idx+1]  # 保持维度
                                view_gt = gt_next_obs[view_key].to(self.model.device)
                                
                                padded_view_pred = view_pred
                                padded_view_gt = view_gt
                                
                                # 收集所有GPU的预测结果
                                gathered_view_pred = accelerator.gather(padded_view_pred)
                                gathered_view_gt = accelerator.gather(padded_view_gt)
                                
                                if accelerator.is_main_process:
                                    view_obs_preds[view_key].append(gathered_view_pred)
                                    view_obs_gts[view_key].append(gathered_view_gt)

                    if accelerator.is_main_process:
                        pepoch.update(1)

                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    # 处理动作预测评估（只在action或joint模式下）
                    if should_predict_actions and len(all_preds) > 0:
                        all_preds = torch.cat(all_preds, dim=0)
                        all_actions = torch.cat(all_actions, dim=0).to(all_preds.device)

                        l1 = torch.nn.functional.l1_loss(all_preds, all_actions)
                        mse = torch.nn.functional.mse_loss(all_preds, all_actions)

                        step_log['val_action_mse'] = mse.item()
                        step_log['val_action_l1'] = l1.item()

                        # 评估每个动作维度
                        action_names = ['lx','ly','lz','lx_rot', 'ly_rot', 'lz_rot', 'l_gripper', 'rx','ry','rz','rx_rot', 'ry_rot', 'rz_rot', 'r_gripper']
                        if use_quaternion:
                            action_names = ['lx','ly','lz','lw_rot', 'lx_rot', 'ly_rot', 'lz_rot', 'l_gripper', 'rx','ry','rz','rw_rot', 'rx_rot', 'ry_rot', 'rz_rot', 'r_gripper']
                        
                        all_actions_dim = all_actions.shape[-1]
                        for i in range(all_actions_dim):
                            if i < len(action_names):
                                l1 = torch.nn.functional.l1_loss(all_preds[:,:,i], all_actions[:,:,i])
                                step_log[f"action/{action_names[i]}"] = l1.item()
                    
                    # 处理图像预测评估（只在next_obs或joint模式下）
                    if should_generate_images:
                        all_views_mse = 0
                        all_views_l1 = 0
                        views_count = 0
                        
                        for view_key in self.rgb_keys:
                            if len(view_obs_preds[view_key]) > 0:
                                views_count += 1
                                view_preds = torch.cat(view_obs_preds[view_key], dim=0) 
                                view_gts = torch.cat(view_obs_gts[view_key], dim=0).to(view_preds.device)
                                
                                view_preds = view_preds[:4] # 只取4个样本，避免内存不足
                                view_gts = view_gts[:4] # 只取4个样本，避免内存不足
                                
                                print(f"原始形状: preds (latent) {view_preds.shape}, gts (raw) {view_gts.shape}")
                                
                                unwrap_model = getattr(self.model, "module", self.model)
                                try:
                                    # 解码预测的潜在向量为图像
                                    decoded_view_preds = unwrap_model.obs_encoder.apply_vae(view_preds, inverse=True)
                                    print(f"VAE解码后形状 (decoded_view_preds): {decoded_view_preds.shape}")

                                    final_decoded_preds = decoded_view_preds

                                    # 为GT图像创建和应用VideoTransform
                                    view_gts = view_gts.float()

                                    obs_transform_eval = VideoTransform(
                                        resize_shape=unwrap_model.obs_encoder.obs_transform.resize_shape,
                                        crop_shape=unwrap_model.obs_encoder.obs_transform.crop_shape,
                                        random_crop=False,
                                        color_jitter=None,
                                        imagenet_norm=unwrap_model.obs_encoder.obs_transform.imagenet_norm
                                    )

                                    view_gts_for_transform = view_gts
                                    transformed_gts = obs_transform_eval(view_gts_for_transform)
                                    final_view_gts = transformed_gts.unsqueeze(1)

                                    print(f"最终预测维度: {final_decoded_preds.shape}, GT维度: {final_view_gts.shape}")

                                    # 确保维度匹配
                                    assert final_decoded_preds.shape == final_view_gts.shape, f'维度不匹配: {final_decoded_preds.shape} != {final_view_gts.shape}'

                                    if final_decoded_preds.shape == final_view_gts.shape:
                                        view_l1 = torch.nn.functional.l1_loss(final_decoded_preds, final_view_gts)
                                        view_mse = torch.nn.functional.mse_loss(final_decoded_preds, final_view_gts)
                                        
                                        step_log[f'image/val_image_mse_{view_key}'] = view_mse.item()
                                        step_log[f'image/val_image_l1_{view_key}'] = view_l1.item()
                                        
                                        all_views_mse += view_mse.item()
                                        all_views_l1 += view_l1.item()
                                        
                                        samples_to_viz = min(4, final_decoded_preds.shape[0])
                                        decoded_imgs_to_plot = self._decode_and_plot_improved(final_decoded_preds[:samples_to_viz])
                                        gt_imgs_to_plot = self._decode_and_plot_improved(final_view_gts[:samples_to_viz])
                                        
                                        step_log[f'image_vis/pred_images_{view_key}'] = wandb.Image(decoded_imgs_to_plot)
                                        step_log[f'image_vis/gt_images_{view_key}'] = wandb.Image(gt_imgs_to_plot)
                                    else:
                                        print(f"维度不匹配，无法计算损失: final_preds {final_decoded_preds.shape}, final_gts {final_view_gts.shape}")
                                except Exception as e:
                                    print(f"图像处理或损失计算出错 for view {view_key}: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        # 记录所有视角的平均损失
                        if views_count > 0:
                            step_log['image/val_image_mse_avg'] = all_views_mse / views_count
                            step_log['image/val_image_l1_avg'] = all_views_l1 / views_count
                    
                    # 添加训练目标信息到日志
                    step_log['predict_target'] = self.train_target
                    if should_predict_actions:
                        step_log['predicted_actions'] = True
                    if should_generate_images:
                        step_log['predicted_images'] = True

    def _decode_and_plot_improved(self, images):
        """改进的图像解码和可视化函数.
        Input 'images' shape: (Batch_viz, View, Channel, Time, Height, Width)
        """
        # 1. 规范化图像值到[0,1]范围
        if images.numel() > 0 and (torch.min(images) < 0 or torch.max(images) > 1): # Check numel to avoid error on empty tensor
            images = (images - torch.min(images)) / (torch.max(images) - torch.min(images) + 1e-8)
        
        b_viz, v_dim, c_dim, t_dim, h_dim, w_dim = images.shape

        if b_viz == 0: # Handle empty input
            return torch.empty(0)

        # We want each row in the grid to be one full time sequence for a given (original batch item, view).
        # 'images' here is already sliced by samples_to_viz (b_viz).
        # Rearrange for make_grid: (TotalFrames, C, H, W)
        # TotalFrames = b_viz * v_dim * t_dim
        images_for_grid = rearrange(images, "b v c t h w -> (b v t) c h w")
        
        # nrow should be t_dim to make each row display one full time sequence for one (b_viz_sample, v_dim_sample)
        # If v_dim > 1, one original sample might span multiple rows in the make_grid output if we only use t_dim as nrow.
        # To have each row be "one item" (sample, view), and columns be time:
        # The input to make_grid should be (NumItems * Time, C, H, W)
        # And nrow = Time
        # NumItems = b_viz * v_dim
        
        # Current images_for_grid is (b_viz * v_dim * t_dim, c_dim, h_dim, w_dim)
        # nrow=t_dim will lay out sequences correctly.
        
        images_grid = make_grid(images_for_grid, nrow=t_dim, padding=2, normalize=False)
        
        # 4. 增加清晰度
        h_new, w_new = images_grid.shape[1] * 2, images_grid.shape[2] * 2
        images_grid_resized = F.interpolate(
            images_grid.unsqueeze(0), 
            size=(h_new, w_new), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return images_grid_resized

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        
        # resume training
        if cfg.training.resume:
            origin_output_dir = self._output_dir
            resume_ckpt_path = default(cfg, 'training.resume_ckpt_path', None)
            force_load = default(cfg, 'training.force_load_checkpoint', False)
            lastest_ckpt_path = self.get_checkpoint_path()
            if resume_ckpt_path is not None:
                resume_ckpt_path = pathlib.Path(resume_ckpt_path)
            elif lastest_ckpt_path.is_file():
                resume_ckpt_path = lastest_ckpt_path
            if resume_ckpt_path.is_file():
                print(f"Resuming from checkpoint {resume_ckpt_path}")
                self.load_checkpoint(path=resume_ckpt_path, exclude_keys=self.exclude_keys, force_load=force_load)
                # set 0 for init new lr
                self.epoch = 0
                self.global_step = 0
            # 不加这个继续训练会把checkpoint,log这些写入到resume_ckpt_path里去
            self._output_dir = origin_output_dir


        # find unsed params
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        fsdp_plugin = accelerate.utils.dataclasses.FullyShardedDataParallelPlugin(sharding_strategy="SHARD_GRAD_OP", backward_prefetch="BACKWARD_PRE", use_orig_params=True)
        accelerator = Accelerator(
            # dispatch_batches=False,
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs],
            fsdp_plugin=fsdp_plugin if self.fsdp else None,
            # 添加以下参数来禁用 DeepSpeed
            deepspeed_plugin=None
        )
        # 验证训练目标设置
        print_rank0(accelerator, f"Training with target: {self.train_target}")
        if self.train_target == 'action':
            print_rank0(accelerator, "Will only train action prediction (video generation frozen)")
        elif self.train_target == 'next_obs':
            print_rank0(accelerator, "Will only train video generation (action prediction frozen)")
        elif self.train_target == 'joint':
            print_rank0(accelerator, "Will train both action prediction and video generation")
        # accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulate_every)
        print_rank0(accelerator, f'cfg.training.gradient_accumulate_every:{cfg.training.gradient_accumulate_every}')
        
        rank = accelerator.process_index
        world_size=accelerator.num_processes
        batch_size=cfg.train_dataloader.batch_size
        
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            default(cfg, 'task.dataset.sample_ratio', 0.1)
            cfg.logging.project = 'diffusion_policy_debug'

        action_dim = cfg.task.shape_meta['action'].shape[0]
        parse_head_action = default(cfg, 'task.parse_head_action', False) 
        parse_head_action_v2 = default(cfg, 'task.parse_head_action_v2', 0) # 0： 不解析，1：x+0.15,z+0.2, 同时解析head_yaw,head_pitch, 2：只x+0.15,z+0.2 
        train_test_split = default(cfg, 'task.dataset.train_val_split', 0.9)
        print(f'parse_head_action:{parse_head_action}, parse_head_action_v2:{parse_head_action_v2}')
        use_quaternion = default(cfg, 'task.use_quaternion', False) # 是否使用四元数表示旋转
        relative_action = default(cfg, 'task.relative_action', False) # 是否使用相对动作
        print(f'use_quaternion:{use_quaternion}, relative_action:{relative_action}')
        trim_stationary = default(cfg, 'task.trim_stationary', False) # 是否去除静止动作
        # configure dataset
        low_dim_obs_horizon = default(cfg, 'task.low_dim_obs_horizon', 1)
        img_obs_horizon = default(cfg, 'task.img_obs_horizon', 1)
        horizon = default(cfg, 'task.action_horizon', 20)
        is_bi_mode = default(cfg, 'task.dataset.is_bi_mode', True)
        action_history_length = default(cfg, 'task.action_history_length', 0)
        image_history_length = default(cfg, 'task.image_history_length', 0)
        is_binocular = default(cfg, 'task.is_binocular', False) # 是否是双目模式
        # is_factory = default(cfg, 'task.is_factory', False) # 是否是工厂模式, 已经被抛弃，改成自动根据工厂手持式设备标号判定是否要加，TODO：未来需要配置化
        relative_action = default(cfg, 'task.relative_action', False) # 是否使用相对动作
        add_noise = default(cfg, 'task.add_noise', False) # 是否添加噪声, 只对相对动作有效
        filter_angle_outliers = default(cfg, 'task.filter_angle_outliers', True) # 是否过滤角度异常值, 默认要过滤
        sample_rate = default(cfg, 'task.dataset.sample_rate', 1.0) # 针对action和image的采样率
        save_meta_data = default(cfg, 'task.dataset.save_meta_data', True) # 是否保存meta数据
        force_overwrite = default(cfg, 'task.dataset.force_overwrite', False) # 是否强制覆盖
        use_gripper_cur = default(cfg, 'task.use_gripper_cur', False) # 是否使用关节力矩, 注意这里默认只使用最后一个关节的力矩
        use_joint_cur = default(cfg, 'task.use_joint_cur', False) # 是否使用观测到的所有关节的力矩
        root_dir = default(cfg, 'task.dataset.root_dir', '/x2robot/Data/.cache/hf_datasets') # 数据集根目录
        predict_next_obs_every_sample = default(cfg, 'training.predict_next_obs_every_sample', 0) # 每多少个样本预测一次下一帧图像
        cam_mapping = _CAM_MAPPING
        # 过滤掉不在rgb_keys里的cam
        filter_cam_mapping = {}
        for key,value in cam_mapping.items():
            if value in self.rgb_keys:
                filter_cam_mapping[key] = value
        cam_mapping = filter_cam_mapping
        merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
        merge_image_history = image_history_length > 0 # obs_history里是否加入图像历史
        print(f'merge_cur_history:{merge_cur_history}, merge_image_history:{merge_image_history}')
        _ACTION_KEY_FULL_MAPPING_XY = {
            'follow_right_arm_joint_pos': 'follow_right_joint_pos',
            'follow_right_arm_joint_dev': 'follow_right_joint_dev',
            'follow_right_arm_joint_cur': 'follow_right_joint_cur',
            'follow_right_ee_cartesian_pos': 'follow_right_position',
            'follow_right_ee_rotation': 'follow_right_rotation',
            'follow_right_gripper': 'follow_right_gripper',
            'master_right_ee_cartesian_pos': 'master_right_position',
            'master_right_ee_rotation': 'master_right_rotation',
            'master_right_gripper': 'master_right_gripper',
            'follow_left_arm_joint_pos': 'follow_left_joint_pos',
            'follow_left_arm_joint_dev': 'follow_left_joint_dev',
            'follow_left_arm_joint_cur': 'follow_left_joint_cur',
            'follow_left_ee_cartesian_pos': 'follow_left_position',
            'follow_left_ee_rotation': 'follow_left_rotation',
            'follow_left_gripper': 'follow_left_gripper',
            'master_left_ee_cartesian_pos': 'master_left_position',
            'master_left_ee_rotation': 'master_left_rotation',
            'master_left_gripper': 'master_left_gripper',
        }
        full_action_keys_needed = list(_ACTION_KEY_FULL_MAPPING_XY.keys()) # Special use for Xinyuan, Note: If you change any single varaible in data_config, the dataset cache will force regenerate, which cause multi-gpu training conflicts
        prediction_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper']
        obs_action_keys = None # obsered的action和预测的action可能不一样：设置为None则默认一样
        if self.minmax_range_robot == 'arx_joint':
            prediction_action_keys = ['follow_left_arm_joint_pos', 'follow_right_arm_joint_pos'] # Check normalizer for detailed explanation
            full_action_keys_needed = prediction_action_keys
        elif self.minmax_range_robot == 'arx_master':
            prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
            full_action_keys_needed = prediction_action_keys
        elif self.minmax_range_robot == 'arx_master_obs_follow':
            prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
            obs_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper']
            full_action_keys_needed = prediction_action_keys + obs_action_keys
        elif self.minmax_range_robot == 'hand_jaka_pos':
            prediction_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_hand_joint_pos','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_hand_joint_pos']
            full_action_keys_needed = list(_ACTION_KEY_WITH_HAND_MAPPING.keys())
        elif self.minmax_range_robot == 'hand_jaka_joint':
            prediction_action_keys = ['follow_left_arm_joint_pos', 'follow_left_hand_joint_pos','follow_right_arm_joint_pos','follow_right_hand_joint_pos']
            full_action_keys_needed = list(_ACTION_KEY_WITH_HAND_MAPPING.keys())
        if use_gripper_cur or use_joint_cur: # 默认加力控都是从臂的力控
            full_action_keys_needed = full_action_keys_needed + ['follow_left_arm_joint_cur', 'follow_right_arm_joint_cur']
        data_configs = []
        data_folders = []
        print(f'Deep Learning model needs to predict following action_keys:{prediction_action_keys}')
        
        default_instruction = default(cfg, 'task.dataset.instruction', '')
        instruction_path = default(cfg, 'task.dataset.instruction_path', None)
        for dataset_dict in cfg.task.dataset.dataset_paths:
            data_folders.append(dataset_dict['path'])
            # 使用get方法但提供当前default_instruction作为默认值
            default_instruction = dataset_dict.get('instruction', default_instruction)
            data_config = X2RDataProcessingConfig()
            data_config.update(
                cam_mapping=cam_mapping,
                default_instruction=default_instruction,
                instruction_path=instruction_path,
                class_type='x2',
                train_test_split=train_test_split,
                filter_angle_outliers=filter_angle_outliers,
                sample_rate=sample_rate,
                parse_tactile=self.use_tactile,
                action_keys=full_action_keys_needed,
                minmax_range_robot=self.minmax_range_robot,
                trim_stationary=trim_stationary,
            )
            data_configs.append(data_config.as_dict())
        
        return_next_obs_indices = [20] if image_history_length == 0 else [1,5,15,20]  # todo: be configured in yaml

        assert len(return_next_obs_indices) == image_history_length + 1, f"len(return_next_obs_indices):{len(return_next_obs_indices)} image_history_length:{image_history_length}"
        data_chunk_config = X2RDataChunkConfig().update(
            left_padding=True if action_history_length > 0 or image_history_length > 0 else False,
            right_padding=True,
            action_horizon=horizon+1,
            action_history_length=action_history_length,
            image_history_length=image_history_length,
            return_next_obs_indices=return_next_obs_indices,
        )

        train_dataset = IterChunkDataset(
            data_folders,
            data_configs,
            data_chunk_config,
            preload_pool_size = 1,
            num_preloader_threads  = 1,
            max_frame_buffer_size = 2000,
            num_frame_producer_threads = 1,
            force_overwrite=force_overwrite,
            split='train',
            accelerator=accelerator,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            slice_size=batch_size,
            save_meta_data=save_meta_data,
            action_keys=full_action_keys_needed,
            root_dir=pathlib.Path(root_dir),
        )
        total_frames = train_dataset.num_frames
        val_dataset = IterChunkDataset(
            data_folders,
            data_configs,
            data_chunk_config,
            preload_pool_size = 1,
            num_preloader_threads  = 1,
            max_frame_buffer_size = 2000,
            num_frame_producer_threads = 1,
            force_overwrite=force_overwrite,
            split='test',
            accelerator=accelerator,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            slice_size=batch_size,
            save_meta_data=save_meta_data,
            action_keys=full_action_keys_needed,
            root_dir=pathlib.Path(root_dir),
        )
        total_frames_val = val_dataset.num_frames
        # 设置collate_fn
        collate_fn = collate_wrapper(
            collate_type = 'chunking',
            low_dim_obs_horizon=low_dim_obs_horizon,
            img_obs_horizon=img_obs_horizon,
            horizon=horizon,
            action_dim=action_dim,
            is_bi_mode=True,
            sample2instruct=None,
            to_lie_algebra=False,
            sample2imginstruct=None,
            parse_head_action=False,
            mask_type=None,
            mask_keys=None,
            merge_cur_history=merge_cur_history,
            merge_image_history=merge_image_history,
            relative_action=relative_action,
            add_noise=add_noise,
            action_keys=prediction_action_keys,
            agent_pos_keys=obs_action_keys,
            use_gripper_cur=use_gripper_cur,
            use_joint_cur=use_joint_cur
        )


        # 计算train/val step
        global_batch_size = batch_size * world_size
        train_num = int(total_frames // batch_size // accelerator.num_processes)
        val_num = int(total_frames_val // batch_size // accelerator.num_processes)

        # 加载dataloader
        train_dataloader = DynamicDataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             num_workers=1,
                                             gpu_id=rank,
                                             collate_fn=collate_fn,
                                             length=train_num)
        val_dataloader = DynamicDataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           num_workers=1,
                                           gpu_id=rank,
                                           collate_fn=collate_fn,
                                           length=val_num)
        train_dataloader = accelerator.prepare(train_dataloader)
        val_dataloader = accelerator.prepare(val_dataloader)
        
        # if world_size == 1:
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] = str(random.randint(30000, 40000))
        #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        print(f"rank {accelerator.process_index} total_frames:{total_frames} total_frames_val:{total_frames_val} train_num {train_num}, val_num {val_num}",flush=True)
        print(f"rank {accelerator.process_index} batch_size_per_rank {batch_size} global_batch_size {global_batch_size}", flush=True)
        accelerator.wait_for_everyone()
        # setup normalizer
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim, minmax_range_robot=self.minmax_range_robot)
        # obs lowdim
        for key in self.lowdim_keys:
            if key == 'agent_pos' and use_gripper_cur: # 注意这里的特殊判断
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim+2, minmax_range_robot=self.minmax_range_robot, use_gripper_cur_pos=use_gripper_cur)
            elif key == 'agent_pos' and use_joint_cur:
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim+14, minmax_range_robot=self.minmax_range_robot, use_joint_cur=use_joint_cur)
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim, minmax_range_robot=self.minmax_range_robot)
        if self.use_tactile:
            for key in self.tactile_keys:
                normalizer[key] = TactileLinearNormalizer.create_tactile_normalizer(tactile_dim=120)
        self.model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=train_num * cfg.training.num_epochs * world_size,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )


        # configure logging
        wandb_run = None
        if accelerator.is_main_process:
            print(self.model)
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                monitor_gym=False,
                **cfg.logging
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = accelerator.device
        self.model.to(device)

        self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, lr_scheduler
        )
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        batch_size = cfg.train_dataloader.batch_size
        gradient_accumulation_steps = cfg.training.gradient_accumulate_every
        # 添加一些时间的统计信息
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()

            self.train_loop(train_num, rank, accelerator, train_dataloader, lr_scheduler, wandb_run, json_logger, cfg, step_log, local_epoch_idx, gradient_accumulation_steps)
            gc.collect()
            train_dataloader.shutdown()
            train_dataloader.dataset.reset_epoch(local_epoch_idx)

            if (self.epoch % cfg.training.val_every) == 0:
                self.validation_loop(val_num, rank, accelerator, val_dataloader, cfg, step_log)
                gc.collect()
                val_dataloader.shutdown()
                val_dataloader.dataset.reset_epoch(self.epoch)

            if (self.epoch % cfg.training.sample_every) == 0:
                self.predict_loop(val_num, rank, accelerator, val_dataloader, cfg, step_log, use_quaternion=use_quaternion, parse_head_action=parse_head_action, parse_head_action_v2=parse_head_action_v2, relative_action=relative_action)
                gc.collect()
                val_dataloader.shutdown()
                val_dataloader.dataset.reset_epoch(self.epoch)
                
            # checkpoint
            if accelerator.is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0:
                # checkpointing
                model_ddp = self.model
                if not self.fsdp:
                    self.model = unwrap_model(self.model, accelerator)
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
                # recover to ddp model
                self.model = model_ddp
            # ========= eval end for this epoch ==========
            torch.cuda.empty_cache()
            gc.collect()
            # log of last step is combined with validation and rollout
            if accelerator.is_main_process:
                if not safe_wandb_log(wandb_run, step_log, step=self.global_step, log_interval=1):
                    print(f"Failed to log to wandb", flush=True)
                json_logger.log(step_log)
            self.epoch += 1
        
        json_logger.stop()
        accelerator.wait_for_everyone()
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainUWMWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

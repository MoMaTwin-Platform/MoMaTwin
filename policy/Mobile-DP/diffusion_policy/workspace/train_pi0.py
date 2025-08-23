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
import hydra
from lerobot.configs.types import FeatureType, PolicyFeature
import torch
from omegaconf import OmegaConf
import pathlib
import torch.distributed
import copy
import random
import wandb
import tqdm
import time
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.pi0.modeling_pi0_v2 import PI0Policy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from x2robot_dataset.common.data_preprocessing import _CAM_MAPPING
from einops import rearrange, reduce
from x2robot_dataset.dataloader import DynamicDataLoader
from x2robot_dataset.common.collate_fn import collate_wrapper
from x2robot_dataset.common.data_utils import decode_text, relative_to_actions
from x2robot_dataset.lazy_dataset import (
    IterChunkDataset,
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dataloader import DynamicDataLoader
from lerobot.configs.policies import PreTrainedConfig
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin

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
    # model = model._orig_mod if is_compiled_module(model) else model
    return model

def convert_model_params_dtype(model, dtype):
    print(f"Converting model parameters to {dtype}")
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return model

class TrainPI0Workspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # 添加梯度裁剪配置，默认为None表示不裁剪
        self.clip_gradnorm = default(cfg, 'clip_gradnorm', None)
        
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
        # configure model
        config = PreTrainedConfig.from_pretrained(cfg.policy)
        config.device = 'cpu'  # Force initial loading on CPU
        attention_implementation = default(cfg, 'training.attention_implementation', 'eager')
        config.attention_implementation = attention_implementation
        config.n_action_steps = cfg.task.action_horizon
        config.chunk_size = cfg.task.action_horizon
        config.num_steps = default(cfg, 'num_train_timesteps', 10)
        self.model: PI0Policy = PI0Policy.from_pretrained(cfg.policy, config=config)

        self.use_fsdp = default(cfg, 'use_fsdp', False)
        param_groups = [
            {'params': self.model.parameters()}
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

    def train_loop(self, train_num, rank, accelerator, train_dataloader, lr_scheduler, wandb_run, json_logger, cfg, step_log, local_epoch_idx, gradient_accumulation_steps):
        # ========= train for this epoch ==========
        # set seed for shuffling
        step_loss = 0.0
        # only display on main process
        tepoch = tqdm.tqdm(total=train_num, desc=f"rank {rank} - Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process)
        self.model.train()
        # t0 = time()
        torch.cuda.nvtx.range_push("batch 0")
        torch.cuda.nvtx.range_push("data load")
        for batch_idx, batch in enumerate(train_dataloader):
            torch.cuda.nvtx.range_pop()
            if batch_idx >= train_num-1:
                print(f"rank {rank} batch_idx {batch_idx} >= train_num {train_num} -1, break")
                break
            # print_rank0(accelerator, f"batch_idx {batch_idx}, batch instruction: {batch['instruction']}")
            # print_rank0(accelerator, f"decode batch instruction: {decode_text(batch['instruction'])}")
            with accelerator.accumulate(self.model):
                # compute loss
                with torch.cuda.nvtx.range(f"forward compute"):
                    raw_loss, _ = self.model(batch)
                with torch.cuda.nvtx.range(f"backward compute"):
                    accelerator.backward(raw_loss)
                
                # 计算梯度范数
                grad_norm = None
                if self.clip_gradnorm is not None:
                    with torch.cuda.nvtx.range(f"grad_norm_compute"):
                        if accelerator.sync_gradients:
                            # 获取所有模型参数的梯度
                            params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                            if len(params) > 0:
                                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2)
                                accelerator.clip_grad_norm_(params, self.clip_gradnorm)
                
                with torch.cuda.nvtx.range(f"optimizer step"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()

            step_loss = raw_loss.detach().item()
            step_log.update({
                'train_loss': step_loss,
                'global_step': self.global_step,
                'epoch': self.epoch,
                'lr': lr_scheduler.get_last_lr()[0]
            })
            
            # 添加梯度范数到日志
            if grad_norm is not None:
                step_log.update({'grad_norm': grad_norm.item()})
            
            if accelerator.is_main_process and self.global_step % gradient_accumulation_steps == 0:
                # logging
                tepoch.set_postfix(loss=step_loss, refresh=False)
                tepoch.update(1)
                is_last_batch = (batch_idx == (len(train_dataloader)-1))
                if not is_last_batch:
                    if not safe_wandb_log(wandb_run, step_log, step=self.global_step, log_interval=20):
                        print(f"Failed to log to wandb", flush=True)
                    json_logger.log(step_log)
            self.global_step += 1 # add at the last

            torch.cuda.nvtx.range_pop()
            
            if batch_idx <= train_num - 2:
                torch.cuda.nvtx.range_push(f"batch {batch_idx+1}")
                torch.cuda.nvtx.range_push("data load")

    def validation_loop(self, val_num, rank, accelerator, val_dataloader, cfg, step_log):
        self.model.eval()
        with torch.no_grad():
            val_losses = list()
            with tqdm.tqdm(total=val_num, desc=f"rank {rank} - Validation epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as vepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= val_num-1:
                        break
                    raw_loss, _ = self.model(batch)
                    
                    # 修改: 先在本地累积loss，最后再同步一次
                    val_losses.append(raw_loss.detach())
                    if accelerator.is_local_main_process:
                        vepoch.update(1)
                        
            # 修改: 在所有batch处理完后，只进行一次gather操作
            if len(val_losses) > 0:
                val_loss = torch.mean(torch.stack(val_losses))
                gathered_loss = accelerator.gather(val_loss)
                if accelerator.is_main_process:
                    step_log['val_loss'] = gathered_loss.mean().item()
        accelerator.wait_for_everyone()
            
    def predict_loop(self, val_num, rank, accelerator, val_dataloader, cfg, step_log, use_quaternion=False, parse_head_action=False, parse_head_action_v2=False, relative_action=False):
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_actions = []
            
            with tqdm.tqdm(total=val_num, desc=f"rank {rank} - predict epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec, 
                    disable=not accelerator.is_local_main_process) as pepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= val_num-1:
                        break
                    gt_action = batch['action'] 
                    result = self.model(batch, is_predict=True)
                    pred_action = result.to(accelerator.device)
                    if relative_action: # 目前只考虑双臂，不考虑其他自由度
                        agent_pos = batch['obs']['agent_pos'].cpu().numpy()
                        pred_action = pred_action.cpu().numpy() 
                        abs_action = []
                        for apos, paction in zip(agent_pos, pred_action):
                            # print_rank0(accelerator, f"agent_pos: {apos.shape}, pred_action: {paction.shape}")
                            abs_action.append(relative_to_actions(paction, apos[0]))
                        pred_action = torch.tensor(abs_action, device=accelerator.device)
                    gt_action = gt_action.to(accelerator.device)
                    
                    # 确保所有进程的数据大小一致
                    pred_action, gt_action = accelerator.pad_across_processes(
                        [pred_action, gt_action],
                        dim=0
                    )
                    
                    # 收集所有GPU的预测结果
                    gathered_preds = accelerator.gather(pred_action)
                    gathered_actions = accelerator.gather(gt_action)
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        all_preds.append(gathered_preds)
                        all_actions.append(gathered_actions)
                        pepoch.update(1)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process and len(all_preds) > 0:
                    all_preds = torch.cat(all_preds, dim=0)
                    all_actions = torch.cat(all_actions, dim=0).to(all_preds.device)

                    # print(all_preds.device)
                    l1 = torch.nn.functional.l1_loss(all_preds, all_actions)
                    mse = torch.nn.functional.mse_loss(all_preds, all_actions)

                    step_log['val_action_mse'] = mse.item()
                    step_log['val_action_l1'] = l1.item()

                    action_names = ['lx','ly','lz','lx_rot', 'ly_rot', 'lz_rot', 'l_gripper', 'rx','ry','rz','rx_rot', 'ry_rot', 'rz_rot', 'r_gripper']
                    if use_quaternion:
                        action_names = ['lx','ly','lz','lw_rot', 'lx_rot', 'ly_rot', 'lz_rot', 'l_gripper', 'rx','ry','rz','rw_rot', 'rx_rot', 'ry_rot', 'rz_rot', 'r_gripper']
                    if parse_head_action:
                        action_names += ['head_yaw', 'head_pitch']
                    elif parse_head_action_v2:
                        action_names += ['head_yaw', 'head_pitch']
                    all_actions_dim = all_actions.shape[-1]
                    for i in range(all_actions_dim):
                        l1 = torch.nn.functional.l1_loss(all_preds[:,:,i], all_actions[:,:,i])
                        step_log[f"{action_names[i]}"] = l1.item()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        if cfg.training.resume:
            origin_output_dir = self._output_dir
            resume_ckpt_path = default(cfg, 'training.resume_ckpt_path', None)
            lastest_ckpt_path = self.get_checkpoint_path()
            if resume_ckpt_path is not None:
                resume_ckpt_path = pathlib.Path(resume_ckpt_path)
            elif lastest_ckpt_path.is_file():
                resume_ckpt_path = lastest_ckpt_path
            if resume_ckpt_path.is_file():
                print(f"Resuming from checkpoint {resume_ckpt_path}")
                self.load_checkpoint(path=resume_ckpt_path, exclude_keys=self.exclude_keys)
                # set 0 for init new lr
                self.epoch = 0
                self.global_step = 0
            # 不加这个继续训练会把checkpoint,log这些写入到resume_ckpt_path里去
            self._output_dir = origin_output_dir

        # 初始化DDP
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        fsdp_plugin = None
        
        # 添加类型转换代码
        # 根据use_fsdp设置模型的数据类型
        print(f"Setting up model with use_fsdp={self.use_fsdp}")
        target_dtype = torch.bfloat16 if self.use_fsdp else torch.float32
        print(f"Converting all model parameters to {target_dtype}")
        
        # 定义深度递归转换函数
        def convert_module_dtype(module, dtype):
            for child in module.children():
                convert_module_dtype(child, dtype)
            
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype != dtype:
                    # print(f"Converting {param_name} from {param.dtype} to {dtype}")
                    param.data = param.data.to(dtype)
            
            for buf_name, buf in module.named_buffers(recurse=False):
                if hasattr(buf, 'dtype') and buf.dtype != dtype and torch.is_floating_point(buf):
                    buf.data = buf.data.to(dtype)
        
        # 执行深度递归转换
        convert_module_dtype(self.model, target_dtype)
        print("Model conversion completed")
        
        if self.use_fsdp:
            fsdp_plugin = FullyShardedDataParallelPlugin(
                sharding_strategy="SHARD_GRAD_OP",
                backward_prefetch="BACKWARD_PRE",
                use_orig_params=True,
            )
        
        # 直接初始化Accelerator (无需FSDP判断，FSDP配置由命令行参数提供)
        if self.use_fsdp:
            accelerator = Accelerator(
                gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
                fsdp_plugin=fsdp_plugin,
                mixed_precision='bf16',
                kwargs_handlers=[ddp_kwargs]
            )
        else:
            accelerator = Accelerator(
                gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
                mixed_precision=None,  # 使用float32时不使用混合精度
                kwargs_handlers=[ddp_kwargs]
            )
        
        print_rank0(accelerator, f'cfg.training.gradient_accumulate_every:{cfg.training.gradient_accumulate_every}')
        
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(f'rank:{rank}, wordsize:{world_size}')
        batch_size = cfg.train_dataloader.batch_size
        
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
        root_dir = default(cfg, 'task.dataset.root_dir', '/x2robot/ganruyi/.cache/hf_datasets') # 数据集根目录
        cam_mapping = _CAM_MAPPING
        # 过滤掉不在rgb_keys里的cam
        filter_cam_mapping = {}
        for key,value in cam_mapping.items():
            if value in self.rgb_keys:
                filter_cam_mapping[key] = value
        cam_mapping = filter_cam_mapping
        merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
        
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
        minmax_range_robot = default(cfg, 'task.minmax_range_robot', 'arx') # 是否是arx,leju,leju_v2
        obs_action_keys = None # obsered的action和预测的action可能不一样：设置为None则默认一样
        if minmax_range_robot == 'arx_joint':
            prediction_action_keys = ['follow_left_arm_joint_pos', 'follow_right_arm_joint_pos'] # Check normalizer for detailed explanation
        elif minmax_range_robot == 'arx_master':
            prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
        elif minmax_range_robot == 'arx_master_obs_follow':
            prediction_action_keys = ['master_left_ee_cartesian_pos','master_left_ee_rotation','master_left_gripper','master_right_ee_cartesian_pos','master_right_ee_rotation','master_right_gripper'] # Check normalizer for detailed explanation
            obs_action_keys = ['follow_left_ee_cartesian_pos','follow_left_ee_rotation','follow_left_gripper','follow_right_ee_cartesian_pos','follow_right_ee_rotation','follow_right_gripper']
        data_configs = []
        data_folders = []
        print(f'Deep Learning model needs to predict following action_keys:{prediction_action_keys}')
        default_instruction = default(cfg, 'task.dataset.instruction', '')
        instruction_path = default(cfg, 'task.dataset.instruction_path', None)
        for dataset_dict in cfg.task.dataset.dataset_paths:
            data_folders.append(dataset_dict['path'])
            # default_instruction = dataset_dict.get('instruction', default_instruction)
            data_config = X2RDataProcessingConfig()
            data_config.update(
                cam_mapping=cam_mapping,
                default_instruction=default_instruction,
                class_type='x2',
                train_test_split=train_test_split,
                filter_angle_outliers=filter_angle_outliers,
                sample_rate=sample_rate,
                parse_tactile=self.use_tactile,
                action_keys=full_action_keys_needed,
                instruction_path=instruction_path,
            )
            data_configs.append(data_config.as_dict())
        
        data_chunk_config = X2RDataChunkConfig().update(
            left_padding=True if action_history_length > 0 else False,
            right_padding=True,
            action_horizon=horizon+1,
            action_history_length=action_history_length,
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
            action_keys=prediction_action_keys,
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
            action_keys=prediction_action_keys,
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
        
        print(f"rank {accelerator.process_index} total_frames:{total_frames} total_frames_val:{total_frames_val} train_num {train_num}, val_num {val_num}",flush=True)
        print(f"rank {accelerator.process_index} batch_size_per_rank {batch_size} global_batch_size {global_batch_size}", flush=True)
        accelerator.wait_for_everyone()
        # setup normalizer
        minmax_range_robot = default(cfg, 'task.minmax_range_robot', 'arx') # 是否是arx,leju,leju_v2
        normalizer = LinearNormalizer()
        # 获取模型参数的真实数据类型
        model_dtype = next(self.model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")
        normalizer['action'] = SingleFieldLinearNormalizer.create_bi_arm_identity(
            action_dim=self.action_dim,
            dtype=model_dtype,  # 使用获取的实际dtype
            minmax_range_robot=minmax_range_robot
        )
        # obs lowdim
        agent_pos_dim = self.action_dim
        for key in self.lowdim_keys:
            if key == 'agent_pos' and use_gripper_cur:
                agent_pos_dim = self.action_dim+2
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(
                    action_dim=agent_pos_dim, 
                    dtype=model_dtype,  # 使用获取的实际dtype 
                    minmax_range_robot=minmax_range_robot, 
                    use_gripper_cur_pos=use_gripper_cur
                )
            elif key == 'agent_pos' and use_joint_cur:
                agent_pos_dim = self.action_dim+14
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(
                    action_dim=agent_pos_dim, 
                    dtype=model_dtype,  # 使用获取的实际dtype
                    minmax_range_robot=minmax_range_robot, 
                    use_joint_cur=use_joint_cur
                )
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(
                    action_dim=agent_pos_dim, 
                    dtype=model_dtype,  # 使用获取的实际dtype
                    minmax_range_robot=minmax_range_robot
                )
        print(f'normalizer:{normalizer}')
        # setup model config
        for key in self.rgb_keys:
            self.model.config.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(224, 224, 3))
        robot_state_feature = PolicyFeature(type=FeatureType.STATE, shape=(agent_pos_dim,))
        self.model.config.input_features['agent_pos'] = robot_state_feature
        self.model.config.output_features['action'] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        
        # 在FSDP之前保存原始的normalizer
        self.normalizer = normalizer
        self.model.set_normalizer(normalizer)

        if self.use_tactile:
            for key in self.tactile_keys:
                normalizer[key] = TactileLinearNormalizer.create_tactile_normalizer(tactile_dim=120)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=train_num * cfg.training.num_epochs * world_size,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            # ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema = hydra.utils.instantiate(cfg.ema, parameters=self.ema_model.parameters())

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

        # device = accelerator.device
        # self.model.to(device)

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
                if not self.use_fsdp:
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
    workspace = TrainPI0Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

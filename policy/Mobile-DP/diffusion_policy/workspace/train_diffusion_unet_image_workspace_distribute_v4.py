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
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import tqdm
import time
import accelerate
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from einops import rearrange, reduce
from x2robot_dataset.common.utils import print_rank_0
from x2robot_dataset.lazy_dataset import (
    X2RDataChunkConfig,
    X2RDataProcessingConfig,
)
from x2robot_dataset.dynamic_robot_dataset import DynamicRobotDataset


OmegaConf.register_new_resolver("eval", eval, replace=True)


def default(config: OmegaConf, attribute_level: str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)


def compute_loss(pred, target):
    loss = F.mse_loss(pred, target, reduction="none")
    loss = loss.type(loss.dtype)
    loss = reduce(loss, "b ... -> b (...)", "mean")
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


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # if not os.path.exists(output_dir):
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # configure
        self.rgb_keys = list()
        self.lowdim_keys = list()
        self.tactile_keys = list()
        obs_shape_meta = cfg.task.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                self.rgb_keys.append(key)
            elif type == "low_dim":
                self.lowdim_keys.append(key)
            elif type == "tactile":
                self.tactile_keys.append(key)
        print_rank_0(f"rgb_keys: {self.rgb_keys}")
        print_rank_0(f"lowdim_keys: {self.lowdim_keys}")
        print_rank_0(f"tactile_keys: {self.tactile_keys}")
        self.use_tactile = len(self.tactile_keys) > 0
        self.action_dim = cfg.task.shape_meta["action"].shape[0]
        self.obs_action_dim = cfg.task.shape_meta["obs"]["agent_pos"].shape[0] if "agent_pos" in cfg.task.shape_meta["obs"] else 0
        self.low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
        self.img_obs_horizon = cfg.task.img_obs_horizon
        self.logging_external = default(cfg, "logging.logging_external", False)

        
        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        # if visual instruct information is provided in cfg.task, then register it to the policy
        # How to add visual instruct ? see diffusion_policy/model/visual_instructors/how_to_use.txt
        if 'visual_instruct' in cfg.task:
            visual_instructor_name = cfg.task.visual_instruct._target_
            print(f"Registering visual instruct {visual_instructor_name} to policy.")
            visual_instructor = hydra.utils.instantiate(cfg.task.visual_instruct)
            self.model.register_visual_instruct(visual_instructor)
        else:
            print("No visual instruct information found in cfg.task.")

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        obs_encorder_lr = cfg.optimizer.lr
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        param_groups = [{"params": self.model.model.parameters()}, {"params": obs_encorder_params, "lr": obs_encorder_lr}]
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")
        self.optimizer = torch.optim.AdamW(params=param_groups, **optimizer_cfg)
        save_optimizer = default(cfg, "checkpoint.save_optimizer", False)
        if not save_optimizer:
            self.exclude_keys = ["optimizer"]

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.fsdp = False

    def train_loop(
        self, train_num, rank, accelerator, dataset, lr_scheduler, wandb_run, json_logger, cfg, step_log, local_epoch_idx, gradient_accumulation_steps
    ):
        # ========= train for this epoch ==========
        train_dataloader = dataset.get_train_dataloader()
        # set seed for shuffling
        step_loss = 0.0
        # only display on main process
        tepoch = tqdm.tqdm(
            total=len(train_dataloader),
            desc=f"rank {rank} - Training epoch {self.epoch}",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
            disable=not accelerator.is_local_main_process,
        )
        self.model.train()
        # t0 = time()
        torch.cuda.nvtx.range_push("batch 0")
        torch.cuda.nvtx.range_push("data load")
        for batch_idx, batch in enumerate(train_dataloader):
            torch.cuda.nvtx.range_pop()
            with accelerator.accumulate(self.model):
                # compute loss
                with torch.cuda.nvtx.range(f"forward compute"):
                    pred, target = self.model(batch)
                    raw_loss = compute_loss(pred, target)
                with torch.cuda.nvtx.range(f"backward compute"):
                    accelerator.backward(raw_loss)
                with torch.cuda.nvtx.range(f"optimizer step"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()

            step_loss = raw_loss.detach().item()
            step_log.update({"train_loss": step_loss, "global_step": self.global_step, "epoch": self.epoch, "lr": lr_scheduler.get_last_lr()[0], "num_train_iters": train_dataloader.max_iters})

            if accelerator.is_main_process and self.global_step % gradient_accumulation_steps == 0:
                # logging
                tepoch.set_postfix(loss=step_loss, refresh=False)
                tepoch.update(1)
                is_last_batch = batch_idx == (len(train_dataloader) - 1)
                if not is_last_batch:
                    if not safe_wandb_log(wandb_run, step_log, step=self.global_step, log_interval=20):
                        print(f"Failed to log to wandb", flush=True)
                    json_logger.log(step_log)
            self.global_step += 1  # add at the last

            torch.cuda.nvtx.range_pop()

            if batch_idx <= train_num - 2:
                torch.cuda.nvtx.range_push(f"batch {batch_idx+1}")
                torch.cuda.nvtx.range_push("data load")

    def validation_loop(self, val_num, rank, accelerator, dataset, cfg, step_log):
        val_dataloader = dataset.get_val_dataloader()
        self.model.eval()
        with torch.no_grad():
            val_losses = list()
            with tqdm.tqdm(
                total=len(val_dataloader),
                desc=f"rank {rank} - Validation epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
                disable=not accelerator.is_local_main_process,
            ) as vepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    pred, target = self.model(batch)
                    raw_loss = compute_loss(pred, target)

                    # 修改: 先在本地累积loss，最后再同步一次
                    val_losses.append(raw_loss.detach())
                    if accelerator.is_local_main_process:
                        vepoch.update(1)

            # 修改: 在所有batch处理完后，只进行一次gather操作
            if len(val_losses) > 0:
                val_loss = torch.mean(torch.stack(val_losses))
                gathered_loss = accelerator.gather(val_loss)
                if accelerator.is_main_process:
                    step_log["val_loss"] = gathered_loss.mean().item()
        accelerator.wait_for_everyone()

    def predict_loop(
        self,
        val_num,
        rank,
        accelerator,
        dataset,
        cfg,
        step_log,
    ):
        val_dataloader = dataset.get_val_dataloader()
        self.model.eval()
        with torch.no_grad():
            all_preds = []
            all_actions = []

            with tqdm.tqdm(
                total=len(val_dataloader),
                desc=f"rank {rank} - predict epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
                disable=not accelerator.is_local_main_process,
            ) as pepoch:
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= val_num - 1:
                        break
                    gt_action = batch["action"]
                    result = self.model(batch, predict=True)
                    pred_action = result["action_pred"].to(self.model.device)

                    gt_action = gt_action.to(self.model.device)

                    # 确保所有进程的数据大小一致
                    pred_action, gt_action = accelerator.pad_across_processes([pred_action, gt_action], dim=0)

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

                    step_log["val_action_mse"] = mse.item()
                    step_log["val_action_l1"] = l1.item()

                    # 简化action_names的生成 - 根据predict_action_keys动态生成
                    action_names = []
                    for key in cfg.task.predict_action_keys:
                        # 根据action key的类型生成相应的名称
                        if 'ee_cartesian_pos' in key:
                            if 'left' in key:
                                action_names.extend(['lx', 'ly', 'lz'])
                            else:
                                action_names.extend(['rx', 'ry', 'rz'])
                        elif 'ee_rotation' in key:
                            if 'left' in key:
                                if '6D' in key:
                                    action_names.extend(['l1_rot', 'l2_rot', 'l3_rot', 'l4_rot', 'l5_rot', 'l6_rot'])
                                else:
                                    action_names.extend(['lx_rot', 'ly_rot', 'lz_rot'])
                            else:
                                if '6D' in key:
                                    action_names.extend(['r1_rot', 'r2_rot', 'r3_rot', 'r4_rot', 'r5_rot', 'r6_rot'])
                                else:
                                    action_names.extend(['rx_rot', 'ry_rot', 'rz_rot'])
                        elif 'gripper' in key:
                            if 'left' in key:
                                action_names.append('l_gripper')
                            else:
                                action_names.append('r_gripper')
                        elif 'velocity_decomposed' in key:
                            action_names.extend(['base_velocity_x', 'base_velocity_y', 'base_velocity_z'])
                        elif 'height' in key:
                            action_names.extend(['height'])
                        elif 'head_rotation' in key:
                            action_names.extend(['head_rotation_y', 'head_rotation_p'])
                        elif 'hand_joint_pos' in key:
                            if 'left' in key:
                                action_names.extend([f'lhand_{i}' for i in range(19)])
                            else:
                                action_names.extend([f'rhand_{i}' for i in range(19)])
                        elif 'arm_joint_pos' in key:
                            if 'left' in key:
                                action_names.extend([f'ljoint_{i}' for i in range(6)])
                            else:
                                action_names.extend([f'rjoint_{i}' for i in range(6)])
                        # 可以继续添加其他类型的映射...
                    
                    # 计算每个action维度的l1 loss
                    all_actions_dim = all_actions.shape[-1]
                    assert len(action_names) == all_actions_dim, f"action_names length {len(action_names)} != action_dim {all_actions_dim}"
                    
                    for i in range(all_actions_dim):
                        l1 = torch.nn.functional.l1_loss(all_preds[:, :, i], all_actions[:, :, i])
                        step_log[f"val_action_l1/{action_names[i]}"] = l1.item()
                    # TODO: There should be a better way to do this: a action mask on action dim [01234567] ...
                    # Add extra loss for comparing orientation (Compare the theta)
                    # if self.data_config.use_6D_rotation is True:
                    #     real_Lang_left = convert_6D_to_Lang(all_actions[:, :, 3:9])
                    #     real_Lang_right = convert_6D_to_Lang(all_actions[:, :, 12:18])
                    #     pred_Lang_left = convert_6D_to_Lang(all_preds[:, :, 3:9])
                    #     pred_Lang_right = convert_6D_to_Lang(all_preds[:, :, 12:18])
                    # else:
                    #     real_Lang_left = convert_euler_to_Lang(all_actions[:, :, 3:6])
                    #     real_Lang_right = convert_euler_to_Lang(all_actions[:, :, 9:12])
                    #     pred_Lang_left = convert_euler_to_Lang(all_preds[:, :, 3:6])
                    #     pred_Lang_right = convert_euler_to_Lang(all_preds[:, :, 9:12])
                    # l1_left = torch.nn.functional.l1_loss(torch.tensor(pred_Lang_left), torch.tensor(real_Lang_left))
                    # l1_right = torch.nn.functional.l1_loss(torch.tensor(pred_Lang_right), torch.tensor(real_Lang_right))
                    # step_log["l1_left_Langle"] = l1_left.item()
                    # step_log["l1_right_Langle"] = l1_right.item()

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # resume training
        if cfg.training.resume:
            origin_output_dir = self._output_dir
            resume_ckpt_path = default(cfg, "training.resume_ckpt_path", None)
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
        fsdp_plugin = accelerate.utils.dataclasses.FullyShardedDataParallelPlugin(
            sharding_strategy="SHARD_GRAD_OP", backward_prefetch="BACKWARD_PRE", use_orig_params=True
        )
        accelerator = Accelerator(
            # dispatch_batches=False,
            gradient_accumulation_steps=cfg.training.gradient_accumulate_every,
            kwargs_handlers=[ddp_kwargs],
            fsdp_plugin=fsdp_plugin if self.fsdp else None,
            # 添加以下参数来禁用 DeepSpeed
            deepspeed_plugin=None,
        )
        print_rank_0(f"cfg.training.gradient_accumulate_every:{cfg.training.gradient_accumulate_every}")
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        print(f"rank:{rank}, wordsize:{world_size}")
        batch_size = cfg.train_dataloader.batch_size

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            default(cfg, "task.dataset.sample_ratio", 0.1)
            cfg.logging.project = "diffusion_policy_debug"

        train_test_split = default(cfg, "task.dataset.train_val_split", 0.9)

        # configure dataset
        horizon = default(cfg, "task.action_horizon", 20)
        action_history_length = default(cfg, "task.action_history_length", 0)
        image_history_length = default(cfg, "task.image_history_length", 0)
        trim_stationary = default(cfg, 'task.trim_stationary', False) # 是否去除静止动作
        filter_angle_outliers = default(cfg, "task.filter_angle_outliers", True)  # 是否过滤角度异常值, 默认要过滤
        cache_dir = default(cfg, "task.dataset.cache_dir", "/x2robot/Data/.cache/dataset_cache")  # 数据集根目录
        dataset_config_path = default(cfg, "task.task_config_path", None)  # 数据集配置文件路径
        assert dataset_config_path is not None, f"dataset_config_path is None, please check your config file"
        
        default_instruction = default(cfg, 'task.dataset.instruction', '')
        instruction_path = default(cfg, 'task.dataset.instruction_path', None)
        instruction_key = default(cfg, 'task.dataset.instruction_key', None)
        one_by_one_relative = default(cfg, 'task.dataset.one_by_one_relative', False)
        
        print(f"instruction_key配置: {instruction_key}")
        print(f"instruction_path配置: {instruction_path}")
        
        # 从shape_meta中构建cam_mapping - 配置化方式
        # camera_name -> obs_key
        cam_mapping = {}
        obs_shape_meta = cfg.task.shape_meta["obs"]
        
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                camera_name = attr.get("camera_name", None)
                if camera_name is not None:
                    cam_mapping[camera_name] = key
                    print_rank_0(f"Added cam mapping: {camera_name} -> {key}")
                else:
                    print_rank_0(f"Warning: RGB observation {key} missing camera_name")

        
        print_rank_0(f"Final cam_mapping: {cam_mapping}")
        merge_cur_history = action_history_length > 0  # agent_pos里是否加入动作历史
        merge_image_history = image_history_length > 0  # 观测图像里是否加入图像历史

        # 直接从任务配置中获取action keys
        predict_action_keys = cfg.task.predict_action_keys
        obs_action_keys = cfg.task.obs_action_keys
        
        # 验证配置
        assert predict_action_keys is not None, "predict_action_keys must be configured in task config"
        assert obs_action_keys is not None, "obs_action_keys must be configured in task config"
        
        print_rank_0(f"Deep Learning model needs to predict following action_keys: {predict_action_keys}")
        print_rank_0(f"Deep Learning model needs to observe following action_keys: {obs_action_keys}")
        
        # 数据配置
        data_config = X2RDataProcessingConfig()
        data_config.update(
            cam_mapping=cam_mapping,
            class_type="x2",
            train_test_split=train_test_split,
            filter_angle_outliers=filter_angle_outliers,
            parse_tactile=self.use_tactile,
            predict_action_keys=predict_action_keys,  # 直接使用配置
            obs_action_keys=obs_action_keys,          # 直接使用配置
            trim_stationary=trim_stationary,
            cache_dir=cache_dir,
            default_instruction=default_instruction,
            instruction_path=instruction_path,
            instruction_key=instruction_key,
            one_by_one_relative = one_by_one_relative,
        )
        data_chunk_config = X2RDataChunkConfig().update(
            left_padding=True if action_history_length > 0 else False,
            right_padding=True,
            predict_action_keys=predict_action_keys,
            action_horizon=horizon,
            action_history_length=action_history_length,
            image_history_length=image_history_length,
            merge_cur_history=merge_cur_history,
            merge_image_history=merge_image_history,
        )

        dataset = DynamicRobotDataset(
            dataset_config_path=dataset_config_path,
            data_config=data_config,
            data_chunk_config=data_chunk_config,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            batch_size=batch_size,
        )
        train_num = dataset.global_train_iters.value
        val_num = dataset.global_val_iters.value
        total_frames = train_num * batch_size * accelerator.num_processes
        total_frames_val = val_num * batch_size * accelerator.num_processes
        # 计算train/val step
        global_batch_size = batch_size * world_size

        print(
            f"rank {accelerator.process_index} total_frames:{total_frames} total_frames_val:{total_frames_val} train_num {train_num}, val_num {val_num}",
            flush=True,
        )
        print(f"rank {accelerator.process_index} batch_size_per_rank {batch_size} global_batch_size {global_batch_size}", flush=True)
        accelerator.wait_for_everyone()
        # setup normalizer
        normalizer = LinearNormalizer()
        
        # action normalizer - 使用predict_action_keys
        normalizer["action"] = SingleFieldLinearNormalizer.create_bi_arm_identity(
            action_dim=self.action_dim, 
            action_keys=predict_action_keys
        )

        if self.obs_action_dim > 0: # Relative Action does not need agent_pos
            normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_bi_arm_identity(
                action_dim=self.obs_action_dim,
                action_keys=obs_action_keys)
        
        # 触觉数据normalizer保持不变
        if self.use_tactile:
            for key in self.tactile_keys:
                normalizer[key] = TactileLinearNormalizer.create_tactile_normalizer(tactile_dim=120)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=train_num * cfg.training.num_epochs * world_size,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, parameters=self.ema_model.parameters())

        # configure logging
        wandb_run = None
        if accelerator.is_main_process:
            print(self.model)
            assert self.logging_external is not None, "Please set cfg.logging_external to True or False"
            if not self.logging_external:
                wandb_run = wandb.init(dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), monitor_gym=False, **cfg.logging)
            else:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    settings=wandb.Settings(
                        disable_job_creation=True,
                        disable_code=True,
                        disable_git=True,
                        save_code=False,
                        x_disable_meta=True,
                        x_disable_stats=True,
                        console="off",
                        silent=True,
                        sync_tensorboard=False,
                        ignore_globs=["*"],
                        anonymous="allow",
                    ),
                    **cfg.logging,
                )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)

        device = accelerator.device
        self.model.to(device)
        if self.ema_model is not None:
            ema.to(device)

        self.model, self.optimizer, lr_scheduler = accelerator.prepare(self.model, self.optimizer, lr_scheduler)
        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        json_logger = JsonLogger(log_path)
        json_logger.start()
        gradient_accumulation_steps = cfg.training.gradient_accumulate_every
        # 添加一些时间的统计信息
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            self.train_loop(
                train_num,
                rank,
                accelerator,
                dataset,
                lr_scheduler,
                wandb_run,
                json_logger,
                cfg,
                step_log,
                local_epoch_idx,
                gradient_accumulation_steps,
            )
            gc.collect()

            if (self.epoch % cfg.training.val_every) == 0:
                self.validation_loop(val_num, rank, accelerator, dataset, cfg, step_log)
                gc.collect()

            if (self.epoch % cfg.training.sample_every) == 0:
                self.predict_loop(
                    val_num,
                    rank,
                    accelerator,
                    dataset,
                    cfg,
                    step_log
                )
                gc.collect()

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
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

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


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

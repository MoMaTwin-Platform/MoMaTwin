if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import gc
import json
import math
import os
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.iterable_dataset import X2RobotDataset, collate_wrapper, ParallelIterator
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
# from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from einops import rearrange, reduce

OmegaConf.register_new_resolver("eval", eval, replace=True)

def default(config:OmegaConf, attribute_level:str, default_value):
    return OmegaConf.select(config, attribute_level, default=default_value)

def compute_loss1(pred, target, loss_mask):
    # loss = self.forward(batch)
    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss * loss_mask.type(loss.dtype)
    loss = reduce(loss, 'b ... -> b (...)', 'mean')
    loss = loss.mean()
    return loss

def compute_loss(pred, target):
    loss = F.mse_loss(pred, target, reduction='none')
    loss = loss.type(loss.dtype)
    loss = reduce(loss, 'b ... -> b (...)', 'mean')
    loss = loss.mean()
    return loss


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
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
        obs_shape_meta = cfg.task.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                self.rgb_keys.append(key)
            elif type == 'low_dim':
                self.lowdim_keys.append(key)
        print(self.rgb_keys)
        print(self.lowdim_keys)
        self.action_dim = cfg.task.shape_meta['action'].shape[0]
        # self.n_obs_steps = cfg.n_obs_steps
        self.low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
        self.img_obs_horizon = cfg.task.img_obs_horizon
        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        
        if cfg.training.freeze_encoder:
            self.model.obs_encoder.eval()
            self.model.obs_encoder.requires_grad_(False)
            
        if hasattr(cfg.policy, 'lm_encoder') and cfg.training.freeze_lm_encoder:
            self.model.lm_encoder.eval()
            self.model.lm_encoder.requires_grad_(False)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)


        # # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())
        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        # print(obs_encorder_params)
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )
        # print(f'self.optimizer:{self.optimizer}')
        # do not save optimizers
        save_optimizer = default(cfg, 'checkpoint.save_optimizer', False)
        if not save_optimizer:
            self.exclude_keys = ['optimizer']

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        # resume training
        if cfg.training.resume:
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
    
        # find unsed params
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulate_every,kwargs_handlers=[ddp_kwargs])
        # accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulate_every)
        
        rank = accelerator.process_index
        world_size=accelerator.num_processes
        print(f'rank:{rank}, wordsize:{world_size}')
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


        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        # configure dataset
        mixed_train_ds, num_train_data = X2RobotDataset.make_interleaved_dataset(
            dataset_paths=cfg.task.dataset.dataset_paths,
            repeat_num=default(cfg, 'task.dataset.repeat_num', 1.0),
            is_bi_mode=default(cfg, 'task.dataset.is_bi_mode', True),
            sample_ratio=default(cfg, 'task.dataset.sample_ratio', None),
            split='train',
            # num_parallel_calls=-1,
            obs_keys=self.rgb_keys,
            action_window=default(cfg, 'task.dataset.action_window', 21),
            history_window=default(cfg, 'task.dataset.history_window', 0),
            # instruction_file=default(cfg, 'policy.instruction_file', None),
        )
        mixed_validation_ds, num_validation_data = X2RobotDataset.make_interleaved_dataset(
            dataset_paths=cfg.task.dataset.dataset_paths,
            repeat_num=default(cfg, 'task.dataset.repeat_num', 1.0),
            is_bi_mode=default(cfg, 'task.dataset.is_bi_mode', True),
            sample_ratio=default(cfg, 'task.dataset.sample_ratio', None),
            split='validation',
            # num_parallel_calls=-1,
            obs_keys=self.rgb_keys,
            action_window=default(cfg, 'task.dataset.action_window', 21),
            history_window=default(cfg, 'task.dataset.history_window', 0),
            # instruction_file=default(cfg, 'policy.instruction_file', None),
        )

        # train_dataloader = mixed_train_ds
        # val_dataloader = mixed_validation_ds
        print(f"train data: {num_train_data}, val data: {num_validation_data}, batch size: {batch_size}")
        instruction_file=default(cfg, 'training.instruction_file', None)
        sample2instruct = None
        if instruction_file is not None and os.path.exists(instruction_file):
            sample2instruct = {}
            with open(instruction_file) as infile:
                for line in infile:
                    json_data = json.loads(line)
                    sample_id = json_data['uid'].split('_frame_')[0]
                    if sample_id not in sample2instruct:
                        sample2instruct[sample_id] = json_data['instruction']
        horizon = default(cfg, 'task.action_horizon', 20) 
        train_dataloader = ParallelIterator(mixed_train_ds, 
                                            num_train_data, 
                                            batch_size=batch_size,
                                            num_dp=world_size,
                                            rank=rank,
                                            collate_fn=collate_wrapper(
                                                    rgb_keys=self.rgb_keys,
                                                    # n_obs_steps=self.n_obs_steps,
                                                    low_dim_obs_horizon=self.low_dim_obs_horizon,
                                                    img_obs_horizon=self.img_obs_horizon,
                                                    horizon=horizon,
                                                    action_dim=self.action_dim,
                                                    rank=rank,
                                                    batch_size=batch_size,
                                                    sample2instruct=sample2instruct,
                                                )
        )

        val_batch_size = cfg.val_dataloader.batch_size
        val_dataloader = ParallelIterator(mixed_validation_ds, 
                                            num_validation_data, 
                                            batch_size=val_batch_size,
                                            num_dp=world_size,
                                            rank=rank,
                                            collate_fn=collate_wrapper(
                                                    rgb_keys=self.rgb_keys,
                                                    # n_obs_steps=self.n_obs_steps,
                                                    low_dim_obs_horizon=self.low_dim_obs_horizon,
                                                    img_obs_horizon=self.img_obs_horizon,
                                                    horizon=horizon,
                                                    action_dim=self.action_dim,
                                                    rank=rank,
                                                    batch_size=batch_size,
                                                    sample2instruct=sample2instruct,
                                                )
        )
                                                        
        # setup normalizer
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim)
        # obs lowdim
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim)
        # obs image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        print(f'last_epoch:{self.global_step-1}')
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) * world_size \
                    // cfg.training.gradient_accumulate_every,
            # every gpu do warmup steps
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
                **cfg.logging
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)

        device = accelerator.device
        self.model.to(device)
        if self.ema_model is not None:
            # self.ema_model.to(device)
            ema.to(device)
        # optimizer_to(self.optimizer, device)
        full_val_dataloader = val_dataloader
        self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # training loopcfg
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        batch_size = cfg.train_dataloader.batch_size
        gradient_accumulation_steps = cfg.training.gradient_accumulate_every
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
                
            step_loss = 0.0
            # set seed for shuffling
            # train_dataloader.set_shuffling_seed(local_epoch_idx)
            # only display on main process
            tepoch = tqdm.tqdm(train_dataloader, desc=f"rank {rank} - Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process)
            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                # device transfer is not need, done with accelerate
                # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                with accelerator.accumulate(self.model):
                    # compute loss
                    # method2
                    # raw_loss = self.model(batch)
                    pred, target = self.model(batch)
                    raw_loss = compute_loss(pred, target)
                    
                    # Gather the losses across all processes for logging (if we use distributed training).
                    # avg_loss = accelerator.gather(raw_loss.repeat(batch_size)).mean()
                    # train_loss += avg_loss.item() / gradient_accumulation_steps
                    accelerator.backward(raw_loss)
                    # if accelerator.sync_gradients:
                    #     accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    # step optimizer
                    # if self.global_step % gradient_accumulation_steps == 0: # accelerate will keep track of batch number. https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation   
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model.parameters())

                step_loss = raw_loss.detach().item()
                step_log = {
                    'train_loss': step_loss,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }
                
                if accelerator.is_main_process:
                    # logging
                    tepoch.set_postfix(loss=step_loss, refresh=False)
                    tepoch.update(1)
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                self.global_step += 1 # add at the last
                # train_loss = 0.0

            # run validation
            if (self.epoch % cfg.training.val_every) == 0:
                # self.model.eval()
                self.model.eval()
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"rank {rank} - Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as vepoch:
                        for batch_idx, batch in enumerate(vepoch):
                            # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            # loss = compute_loss(self.model, batch)
                            pred, target,  = self.model(batch)
                            # gather all preds and targets
                            allgpu_pred = accelerator.gather(pred)
                            allgpu_target = accelerator.gather(target)
                            # allgpu_loss_mask = accelerator.gather(loss_mask)
                            raw_loss = compute_loss(allgpu_pred, allgpu_target)
                            # raw_loss = compute_loss(allgpu_pred, allgpu_target, allgpu_loss_mask)
                            # loss = self.model.compute_loss(batch)
                            val_losses.append(raw_loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a val batch : changed
            # if accelerator.is_main_process and (self.epoch % cfg.training.sample_every) == 0:
            if (self.epoch % cfg.training.sample_every) == 0:
                # self.model.eval()
                unwrapped_model = unwrap_model(self.model)
                unwrapped_model.eval()
                # self.model.eval()
                with torch.no_grad():
                    all_preds = []
                    all_actions = []
                    with tqdm.tqdm(full_val_dataloader, desc=f"rank {rank} - predict epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as pepoch:
                        for batch_idx, batch in enumerate(pepoch):
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            instructions = None
                            if hasattr(cfg.policy, 'lm_encoder'):
                                instructions = batch['instruction']

                            # result = self.model.predict_action(obs_dict, instructions)
                            result = unwrapped_model.predict_action(obs_dict, instructions)
                            pred_action = result['action_pred'].to(self.model.device)
                            gt_action = gt_action.to(self.model.device)
                            # print(f'Before rank {rank} - shape of pred_action: {pred_action.shape}, shape of gt_action: {gt_action.shape}, {pred_action}, {gt_action}')
                            # gather all predictions and actions
                            pred_actions = accelerator.gather(pred_action)
                            gt_actions = accelerator.gather(gt_action)
                            # print(f'After rank {rank} - shape of pred_actions: {pred_actions.shape}, shape of gt_actions: {gt_actions.shape}')

                            all_preds.append(pred_actions)
                            all_actions.append(gt_actions)
                        all_preds = torch.cat(all_preds, dim=0)
                        all_actions = torch.cat(all_actions, dim=0).to(all_preds.device)
                        

                        # print(all_preds.device)
                        l1 = torch.nn.functional.l1_loss(all_preds, all_actions)
                        mse = torch.nn.functional.mse_loss(all_preds, all_actions)
                        step_log['val_action_mse'] = mse.item()
                        step_log['val_action_l1'] = l1.item()
                        action_des = ['x','y','z','x_roll','y_pitch','z_yaw','gripper']
                        for i in range(all_actions.shape[-1]):
                            l1 = torch.nn.functional.l1_loss(all_preds[:,:,i], all_actions[:,:,i])
                            prefix = 'l'
                            des_index = i
                            if i >= 7:
                                prefix = 'r'
                                des_index = i - 7
                            step_log[f"{prefix}_{action_des[des_index]}"] = l1.item()
                del unwrapped_model
                # self.model.train()
                
            # checkpoint
            if accelerator.is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0:
                # checkpointing
                model_ddp = self.model
                self.model = unwrap_model(self.model)
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
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
            # self.global_step += 1
            self.epoch += 1
        
        json_logger.stop()
        accelerator.wait_for_everyone()
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

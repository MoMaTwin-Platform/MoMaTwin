if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import gc
import math
import os
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.iterable_dataset import X2RobotDataset, collate_wrapper, FlatIterator
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
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
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
        self.n_obs_steps = cfg.n_obs_steps
        # configure model
        self.model: DiffusionUnetTimmPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetTimmPolicy = None
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
        print(f'obs_encorder params: {len(obs_encorder_params)}')
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

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            print(f'lastest_ckpt_path is {lastest_ckpt_path}')
            # if lastest_ckpt_path.is_file():
            if os.path.isfile(lastest_ckpt_path):
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        batch_size=cfg.train_dataloader.batch_size
        world_size = 1
        rank = 0

        # configure dataset
        mixed_train_ds, num_train_data = X2RobotDataset.make_interleaved_dataset(
            dataset_paths=cfg.task.dataset.dataset_paths,
            repeat_num=default(cfg, 'task.dataset.repeat_num', 1.0),
            is_bi_mode=default(cfg, 'task.dataset.is_bi_mode', True),
            sample_ratio=default(cfg, 'task.dataset.sample_ratio', None),
            split='train',
            from_rawdata=True,
            train_val_split=cfg.task.dataset.train_val_split,
            train_split_seed=cfg.training.seed,
            preload_pool_size = cfg.task.dataset.preload_pool_size,
            num_preloader_threads = cfg.task.dataset.num_preloader_threads,
            max_epoch=cfg.training.num_epochs,
            obs_keys=self.rgb_keys
        )
        mixed_validation_ds, num_validation_data = X2RobotDataset.make_interleaved_dataset(
            dataset_paths=cfg.task.dataset.dataset_paths,
            repeat_num=default(cfg, 'task.dataset.repeat_num', 1.0),
            is_bi_mode=default(cfg, 'task.dataset.is_bi_mode', True),
            sample_ratio=default(cfg, 'task.dataset.sample_ratio', None),
            split='validation',
            from_rawdata=True,
            train_val_split=cfg.task.dataset.train_val_split,
            train_split_seed=cfg.training.seed,
            preload_pool_size = cfg.task.dataset.preload_pool_size,
            num_preloader_threads = cfg.task.dataset.num_preloader_threads,
            max_epoch=cfg.training.num_epochs,
            obs_keys=self.rgb_keys
        )

        train_dataloader = FlatIterator(mixed_train_ds,
                                    batch_size=batch_size,
                                    max_buffersize=cfg.train_dataloader.max_buffersize,
                                    num_processes=cfg.train_dataloader.num_processes,
                                    n_obs_steps=self.n_obs_steps,
                                    num_dp=world_size,
                                    horizon=cfg.horizon,
                                    skip_every_n_frames = cfg.skip_history_every_n_frames,
                                    n_sub_samples=cfg.train_dataloader.n_sub_samples,
                                    rank=rank,
                                    collate_fn=collate_wrapper(
                                            rgb_keys=self.rgb_keys, 
                                            n_obs_steps=self.n_obs_steps, 
                                            horizon=cfg.horizon, 
                                            action_dim=self.action_dim,
                                            rank=rank, 
                                            batch_size=batch_size
                                    )
        )


        val_batch_size = cfg.val_dataloader.batch_size
        val_dataloader = FlatIterator(mixed_validation_ds,
                                    batch_size=val_batch_size,
                                    max_buffersize=cfg.val_dataloader.max_buffersize,
                                    num_processes=cfg.val_dataloader.num_processes,
                                    n_obs_steps=self.n_obs_steps,
                                    num_dp=world_size,
                                    horizon=cfg.horizon,
                                    skip_every_n_frames = cfg.skip_history_every_n_frames,
                                    n_sub_samples=cfg.val_dataloader.n_sub_samples,
                                    rank=rank,
                                    collate_fn=collate_wrapper(
                                            rgb_keys=self.rgb_keys, 
                                            n_obs_steps=self.n_obs_steps, 
                                            horizon=cfg.horizon, 
                                            action_dim=self.action_dim,
                                            rank=rank, 
                                            batch_size=batch_size
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

        assert cfg.training.max_train_steps, "max_train_steps must be set"
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.max_train_steps \
                    // cfg.training.gradient_accumulate_every,
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
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            # self.ema_model.to(device)
            ema.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            if cfg.training.freeze_encoder:
                self.model.obs_encoder.eval()
                self.model.obs_encoder.requires_grad_(False)

            # set seed for shuffling
            # train_dataloader.set_shuffling_seed(local_epoch_idx)
            # train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True), ignore_keys=['instruction'])
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model.parameters())

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    # train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        # 'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                    
                    if batch_idx >= (cfg.training.max_train_steps-1):
                        break

                    # run validation
                    if (batch_idx % cfg.training.val_every_every_n_steps) == 0:
                        self.model.eval()
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                                for val_batch_idx, batch in enumerate(vepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True), ignore_keys=['instruction'])
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and val_batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss
                        self.model.train()

                    # run diffusion sampling on a val batch : changed
                    if (batch_idx % cfg.training.sample_every_n_steps) == 0:
                        self.model.eval()
                        with torch.no_grad():
                            all_preds = []
                            all_actions = []
                            with tqdm.tqdm(val_dataloader, desc=f"predict epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as pepoch:
                                for pred_batch_idx, batch in enumerate(pepoch):
                                    obs_dict = batch['obs']
                                    gt_action = batch['action']
                                    
                                    result = self.model.predict_action(obs_dict)
                                    pred_action = result['action_pred']
                                    all_preds.append(pred_action)
                                    all_actions.append(gt_action)
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
                        self.model.train()
                        torch.cuda.empty_cache()
                        gc.collect()
                
                    # checkpoint
                    if (batch_idx % cfg.training.checkpoint_every_n_steps) == 0:
                        # checkpointing
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
                # ========= eval end for this epoch ==========
                # policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                # self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

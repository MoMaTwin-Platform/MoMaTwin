if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import gc
import os
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import torch.distributed
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import time
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import is_compiled_module
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch.nn.functional as F
from einops import rearrange, reduce
# from x2robot_dataset.data_utils import shuffle_sub_dataset
from x2robot_dataset.map_dataset import (
    make_chunk_dataset,
    collate_wrapper,
)
from torch.utils.data import WeightedRandomSampler
from x2robot_dataset.data_preprocessing import (
    _ACTION_KEY_EE_MAPPING,
    _HEAD_ACTION_MAPPING,
    _HEAD_ACTION_MAPPING_v2,
    _CAM_FILE_MAPPING,
    _CAM_MAPPING,
    _CAM_BINOCULAR_FILE_MAPPING,
    _CAM_BINOCULAR_MAPPING
)
from torch.utils.data import DataLoader
import torch.distributed as dist
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

def safe_wandb_log(wandb_run, step_log, step, max_retries=3):
    for attempt in range(max_retries):
        try:
            wandb_run.log(step_log, step=step)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to log to wandb after {max_retries} attempts: {e}")
                return False
            time.sleep(1 * (attempt + 1))  # 指数退避

global train_dataset
train_dataset = None
def init_worker_train(worker_id):
    global train_dataset
    if train_dataset is not None:
        for dataset in train_dataset.datasets:
            if hasattr(dataset, "load_data"):
                dataset.load_data()

global val_dataset
val_dataset = None
def init_worker_val(worker_id):
    global val_dataset
    if val_dataset is not None:
        for dataset in val_dataset.datasets:
            if hasattr(dataset, "load_data"):
                dataset.load_data()

def print_rank0(accelerator, msg):
    if accelerator.is_main_process:
        print(msg, flush=True)

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
        self.low_dim_obs_horizon = cfg.task.low_dim_obs_horizon
        self.img_obs_horizon = cfg.task.img_obs_horizon
        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        
        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        obs_encorder_lr = cfg.optimizer.lr
        # if cfg.policy.obs_encoder.pretrained:
        #     obs_encorder_lr *= 0.1
        #     print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        # print(obs_encorder_params)
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
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
        
    
        # find unsed params
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulate_every,kwargs_handlers=[ddp_kwargs])
        # accelerator = Accelerator(gradient_accumulation_steps=cfg.training.gradient_accumulate_every)
        print_rank0(accelerator, f'cfg.training.gradient_accumulate_every:{cfg.training.gradient_accumulate_every}')
        
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
        
        action_dim = cfg.task.shape_meta['action'].shape[0]
        parse_head_action = default(cfg, 'task.parse_head_action', False) 
        parse_head_action_v2 = default(cfg, 'task.parse_head_action_v2', 0) # 0： 不解析，1：x+0.15,z+0.2, 同时解析head_yaw,head_pitch, 2：只x+0.15,z+0.2 
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
        obs_keys = list(_CAM_FILE_MAPPING.keys()) if not is_binocular else list(_CAM_BINOCULAR_FILE_MAPPING.keys()) 
        # 过滤掉不在rgb_keys里的cam
        obs_keys = [key for key in obs_keys if key in self.rgb_keys]
        merge_cur_history = action_history_length > 0 # agent_pos里是否加入动作历史 
        
        action_keys = list(_ACTION_KEY_EE_MAPPING.keys())
        if parse_head_action:
            action_keys += list(_HEAD_ACTION_MAPPING.keys())
        elif parse_head_action_v2:
            action_keys += list(_HEAD_ACTION_MAPPING_v2.keys())
        default_data_configs = {'default_instruction': 'spread the clothes',
                            'action_horizon': horizon + low_dim_obs_horizon, # 21,
                            'action_history_length': action_history_length, # 0,
                            'image_horizon': img_obs_horizon,
                            'image_history_length': image_history_length,
                            'right_padding': True,
                            'left_padding': True,
                            'train_val_split': default(cfg, 'task.dataset.train_val_split', 0.9),
                            'split_seed': 42,
                            'obs_keys': obs_keys,
                            'action_keys': action_keys, 
                            }
        # temperally deal with data
        data_configs = []
        sample_ratios = []
        # deal with path
        dataset_paths = []
        for dataset_dict in cfg.task.dataset.dataset_paths:
            dataset_paths.append(dataset_dict['path'])
            data_config = default_data_configs.copy()
            data_config['default_instruction'] = dataset_dict['instruction'] if 'instruction' in dataset_dict else ''
            sample_ratio = dataset_dict['sample_ratio'] if 'sample_ratio' in dataset_dict else 1.0 
            data_configs.append(data_config)
            sample_ratios.append(sample_ratio)
        cache_dir = default(cfg, 'task.dataset.cache_dir', '/x2robot/Data/.cache') 
        flush_cache = default(cfg, 'task.dataset.flush_cache', False)
        filter_angle_outliers = default(cfg, 'task.dataset.filter_angle_outliers', True)

        mask_meta_file_path = default(cfg, 'mask_meta_file_path', None)
        mask_type = default(cfg, 'mask_type', None)
        mask_in_buffer = default(cfg, 'mask_in_buffer', False)
        mask_keys = default(cfg, 'mask_keys', None)
        if mask_keys == 'None':
            mask_keys = None
        if world_size == 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(random.randint(30000, 40000))
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        cam_mapping = _CAM_MAPPING if not is_binocular else _CAM_BINOCULAR_MAPPING 
        # 过滤rgb_keys里没有的cam
        cam_mapping = {k:v for k,v in cam_mapping.items() if v in self.rgb_keys}
        # mixed_train_ds, mixed_validation_ds, train_sampler, val_sampler = make_chunk_dataset(
        global train_dataset
        global val_dataset
        train_dataset, val_dataset, train_weight, val_weight = make_chunk_dataset(
                            dataset_paths,
                            rank=rank,
                            dp_world_size=world_size,
                            cam_mapping=cam_mapping,
                            cache_dir=cache_dir, #缓存数据集的地址，可以被其它进程共享
                            read_labeled_data=False, #是否读取人工标注的数据
                            trans_zh2en_url=None, #翻译人工标注的中文指令，如果缺乏，则不翻译
                            read_from_cache=False, #是否从缓存中加载，如果缓存为空，则先生成缓存
                            # dataset_buffer_size=20, #数据集缓冲区大小
                            memory_size = 10,
                            flush_cache=flush_cache, #是否清空缓存
                            # flush_cache=True, #是否清空缓存
                            data_configs=data_configs, #数据配置
                            num_workers=40, #数据处理的进程数
                            filter_angle_outliers=filter_angle_outliers, #是否平滑角度异常值
                            detect_motion=True, #是否去掉静止不动的样本
                            trim_stationary=False, #是否去掉首尾不动的部分
                            sample_ratio=sample_ratios,
                            parse_head_action=parse_head_action, #是否解析头部动作
                            parse_head_action_v2=parse_head_action_v2, #是否解析全部的头部动作
                            mask_meta_file_path=mask_meta_file_path, # mask文件地址
                            mask_type=mask_type,
                            mask_in_buffer=mask_in_buffer,
                            mask_keys=mask_keys
                            ) 

        train_sampler = WeightedRandomSampler(train_weight, len(train_weight), replacement=False)
        val_sampler = WeightedRandomSampler(val_weight, len(val_weight), replacement=False)
        accelerator.wait_for_everyone()

        collate_fn = collate_wrapper(obs_keys=default_data_configs['obs_keys'],
                                        low_dim_obs_horizon=low_dim_obs_horizon,
                                        img_obs_horizon=img_obs_horizon,
                                        horizon=horizon,
                                        action_dim=action_dim,
                                        is_bi_mode=is_bi_mode,
                                        parse_head_action=parse_head_action,
                                        parse_head_action_v2=parse_head_action_v2,
                                        mask_type=mask_type,
                                        mask_keys=mask_keys,
                                        merge_cur_history=merge_cur_history,
                                        relative_action=relative_action,)
        
        # setup normalizer
        minmax_range_robot = default(cfg, 'task.minmax_range_robot', 'arx') # 是否是arx,leju,leju_v2
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim, minmax_range_robot=minmax_range_robot)
        # obs lowdim
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_bi_arm_identity(action_dim=self.action_dim, minmax_range_robot=minmax_range_robot)
        # obs image, no need to normalize
        # for key in self.rgb_keys:
            # normalizer[key] = get_image_range_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        num_train_ds = len(train_dataset)
        # mixed_train_ds.reset_data(0)
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(num_train_ds * cfg.training.num_epochs) // batch_size // cfg.training.gradient_accumulate_every,
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

        device = accelerator.device
        self.model.to(device)
        if self.ema_model is not None:
            ema.to(device)

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10, collate_fn=collate_fn, worker_init_fn=init_worker_val)
        self.model, self.optimizer, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, val_dataloader, lr_scheduler
        )
        # full_val_dataloader = val_dataloader

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        batch_size = cfg.train_dataloader.batch_size
        gradient_accumulation_steps = cfg.training.gradient_accumulate_every
        # 添加一些时间的统计信息
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            # set seed for shuffling
            # mixed_train_ds.reset_data(epoch_id=local_epoch_idx)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, sampler=train_sampler, collate_fn=collate_fn, worker_init_fn=init_worker_train)
            train_dataloader = accelerator.prepare(train_dataloader)
            # accelerator.wait_for_everyone()
            step_loss = 0.0
            
            # only display on main process
            tepoch = tqdm.tqdm(total=len(train_dataloader)//gradient_accumulation_steps, desc=f"rank {rank} - Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process)
            self.model.train()
            # t0 = time()
            for batch_idx, batch in enumerate(train_dataloader):
                # print(f'action: {batch["action"].shape}, agent_pos: {batch["obs"]["agent_pos"].shape}')
                # device transfer is not need, done with accelerate
                # t1 = time()
                # print_rank0(accelerator, f"batch {batch_idx} load data time: {t1-t0}")
                with accelerator.accumulate(self.model):
                    # compute loss
                    pred, target = self.model(batch)
                    raw_loss = compute_loss(pred, target)
                    # t2 = time()
                    # print_rank0(accelerator, f"batch {batch_idx} forward time: {t2-t1}")
                    # with torch.profiler.profile(
                    #     activities=[
                    #         torch.profiler.ProfilerActivity.CPU,
                    #         torch.profiler.ProfilerActivity.CUDA,
                    #     ]
                    # ) as prof:
                    #     accelerator.backward(raw_loss)
                    #     prof.step()
                    accelerator.backward(raw_loss)
                    # t3 = time()
                    # print_rank0(accelerator, f"batch {batch_idx} backward time: {t3-t2}")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr_scheduler.step()
                    # t4 = time()
                    # print_rank0(accelerator, f"batch {batch_idx} optimizer time: {t4-t3}")
                    # if accelerator.sync_gradients:
                    #     # update ema
                    #     if cfg.training.use_ema:
                    #         ema.step(self.model.parameters())
                    # print_rank0(accelerator, prof.key_averages().table())

                step_loss = raw_loss.detach().item()
                step_log = {
                    'train_loss': step_loss,
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'lr': lr_scheduler.get_last_lr()[0]
                }
                
                if accelerator.is_main_process and self.global_step % gradient_accumulation_steps == 0:
                    # logging
                    tepoch.set_postfix(loss=step_loss, refresh=False)
                    tepoch.update(1)
                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        # wandb_run.log(step_log, step=self.global_step)
                        if not safe_wandb_log(wandb_run, step_log, step=self.global_step):
                            print(f"Failed to log to wandb", flush=True)
                        json_logger.log(step_log)
                self.global_step += 1 # add at the last
                # t5 = time()
                # print_rank0(accelerator, f"other time: {t5-t4}")
                # t0 = time()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0:
                self.model.eval()
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"rank {rank} - Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as vepoch:
                        for batch_idx, batch in enumerate(vepoch):
                            # 所有进程都需要知道这个batch是否有效
                            batch_valid = torch.tensor(1.0 if not (isinstance(batch, dict) and all(v.shape[0] == 0 for v in batch.values())) else 0.0,
                                                    device=accelerator.device)
                            # 同步所有进程，确保所有进程都知道这个batch是否有效
                            batch_valid = accelerator.gather(batch_valid).sum()
                            
                            # 如果所有进程都收到无效batch，跳过这个batch
                            if batch_valid == 0:
                                continue

                            try:
                                pred, target = self.model(batch)
                                raw_loss = compute_loss(pred, target)
                                # 确保所有进程都能得到相同的loss
                                gathered_losses = accelerator.gather(raw_loss)
                                val_losses.append(gathered_losses.mean())
                                
                            except Exception as e:
                                # 记录错误但继续执行
                                if accelerator.is_local_main_process:
                                    print(f"Error processing batch {batch_idx}: {str(e)}")
                                # 通知所有进程发生了错误
                                error_tensor = torch.tensor(1.0, device=accelerator.device)
                                accelerator.gather(error_tensor)
                                continue
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

                accelerator.wait_for_everyone()
            
            if (self.epoch % cfg.training.sample_every) == 0:
                unwrapped_model = unwrap_model(self.model)
                unwrapped_model.eval()
                with torch.no_grad():
                    all_preds = []
                    all_actions = []
                    with tqdm.tqdm(val_dataloader, desc=f"rank {rank} - predict epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec, disable=not accelerator.is_local_main_process) as pepoch:
                        for batch_idx, batch in enumerate(pepoch):
                            # 同步检查batch是否有效
                            batch_valid = torch.tensor(1.0 if not (isinstance(batch, dict) and all(v.shape[0] == 0 for v in batch.values())) else 0.0,
                                                    device=accelerator.device)
                            batch_valid = accelerator.gather(batch_valid).sum()
                            
                            if batch_valid == 0:
                                continue    
                            try:
                                gt_action = batch['action']
                                result = unwrapped_model.predict_action(batch)
                                pred_action = result['action_pred'].to(unwrapped_model.device)
                                gt_action = gt_action.to(unwrapped_model.device)
                                
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
                                    
                            except Exception as e:
                                if accelerator.is_local_main_process:
                                    print(f"Error in prediction batch {batch_idx}: {str(e)}")
                                # 通知所有进程发生了错误
                                error_tensor = torch.tensor(1.0, device=accelerator.device)
                                accelerator.gather(error_tensor)
                                continue

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
                if not safe_wandb_log(wandb_run, step_log, step=self.global_step):
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
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

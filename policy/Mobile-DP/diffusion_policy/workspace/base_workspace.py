import os
import pathlib
import hydra
import copy
import dill
import torch
import yaml
import threading

from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest', 
                        exclude_keys=None,
                        include_keys=None,
                        use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
    
        # 读取 task_config_path 指向的 YAML 文件
        task_config_path = Path(self.cfg.task.task_config_path)
        with task_config_path.open('r') as f:
            task_config = yaml.safe_load(f)

        # 获取 dataset_path
        dataset_path = task_config.get('dataset_path', None)
        if dataset_path is not None:
            # 更新配置中的路径
            self.cfg.task.dataset.dataset_paths = dataset_path
            self.cfg.task.dataset_path = dataset_path
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, force_load=False, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None and 'pickles' in payload:
            include_keys = payload['pickles'].keys()
        elif include_keys is None:
            include_keys = tuple()

        try:
            for key, value in payload['state_dicts'].items():
                if key not in exclude_keys:
                    self.__dict__[key].load_state_dict(value, **kwargs)
            for key in include_keys:
                if key in payload['pickles'] and key not in exclude_keys:
                    self.__dict__[key] = dill.loads(payload['pickles'][key])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            if force_load:
                print("### 💥try harder to squeeze checkpoint weights into new model💥 ###",flush=True)
                new_ckpt = {}
                state_dict = self.__dict__["model"].state_dict()
                for name,param in payload['state_dicts']["model"].items():
                    if name in state_dict.keys():
                        if param.size() == state_dict[name].size():
                            new_ckpt[name] = param
                        else:
                            size_0 = param.size()
                            size_1 = state_dict[name].size()
                            new_ckpt[name] = state_dict[name]
                            slices = [slice(0, min(old_dim, new_dim)) for old_dim, new_dim in zip(size_0, size_1)]
                            new_ckpt[name][slices] = param[slices]
                            print(f"not match key: {name}, required shape: {size_1}, loaded shape: {size_0}, new shape: {new_ckpt[name].size()}")
                    else:
                        print(f"Not used parameter: {name}")
            err = self.__dict__["model"].load_state_dict(new_ckpt, strict=False)
            print("squeeze checkpoint results:", err, flush=True)
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            force_load=False,
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys,
            force_load=force_load)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

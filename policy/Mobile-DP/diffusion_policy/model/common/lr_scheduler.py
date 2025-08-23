from enum import Enum, unique
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_scheduler_with_warmup_and_decay(optimizer, num_warmup_steps, num_training_steps, max_lr, min_lr, decay_steps):
    """
    Custom LR scheduler that warms up to max_lr over num_warmup_steps, then decays to min_lr over decay_steps using a cosine
    decay, and then keeps the LR constant at min_lr for the rest of the training.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer associated with the model parameters.
        num_warmup_steps (int): Number of steps to linearly increase the learning rate to max_lr.
        num_training_steps (int): Total number of training steps.
        max_lr (float): Maximum learning rate during warmup.
        min_lr (float): Minimum learning rate after decay.
        decay_steps (int): Number of steps over which to decay the learning rate from max_lr to min_lr.

    Returns:
        LambdaLR: A PyTorch LR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps)) * (max_lr / optimizer.defaults['lr'])
        elif current_step < num_warmup_steps + decay_steps:
            # Cosine decay
            progress = (current_step - num_warmup_steps) / float(max(1, decay_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed_lr = (max_lr - min_lr) * cosine_decay + min_lr
            return decayed_lr / optimizer.defaults['lr']
        else:
            # Constant minimum learning rate
            return min_lr / optimizer.defaults['lr']

    return LambdaLR(optimizer, lr_lambda)

# @unique
# class ExtendSchedulerType(Enum):
#     # copy old value
#     LINEAR = "linear"
#     COSINE = "cosine"
#     COSINE_WITH_RESTARTS = "cosine_with_restarts"
#     POLYNOMIAL = "polynomial"
#     CONSTANT = "constant"
#     CONSTANT_WITH_WARMUP = "constant_with_warmup"
#     PIECEWISE_CONSTANT = "piecewise_constant"
#     # add new value
#     COSINE_WITH_WARMUP_DECAY = 'cosine_with_warmup_max_decay_min'

# TYPE_TO_SCHEDULER_FUNCTION[ExtendSchedulerType.COSINE_WITH_WARMUP_DECAY] = cosine_scheduler_with_warmup_and_decay

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    print(f'name:{name}')
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

from typing import Dict, List, Optional
from abc import abstractmethod

import tensorflow as tf

REGISTERED_ROBOT_DATASETS = {}
def register_dataset(target_class):
    REGISTERED_ROBOT_DATASETS[target_class.__name__] = target_class

def create_dataset(dataset_name:str, *args, **kwargs):
    """
    Instantiates a robot dataset

    Args:
        robot_dataset (str): the name of the robot dataset to instantiate
        args: arguments passed into the dataset class
        kwargs: keyword arguments passed into the dataset class
    """
    if dataset_name not in REGISTERED_ROBOT_DATASETS:
        raise Exception("Unknown robot dataset: {}, available among{}".format(dataset_name, ", ".join(REGISTERED_ROBOT_DATASETS)))

    return REGISTERED_ROBOT_DATASETS[dataset_name](*args, **kwargs)




class RobotDataMeta(type):
    """Metaclass for registering robot datasets."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_dataset(cls)
        return cls
    
class RobotDataset(object):
    @abstractmethod
    def create_training_dataset(self) -> tf.data.Dataset:
        pass

    @property
    @abstractmethod
    def training_data_nums(self) -> int:
        pass
    
    @abstractmethod
    def create_validation_dataset(self) -> tf.data.Dataset:
        pass


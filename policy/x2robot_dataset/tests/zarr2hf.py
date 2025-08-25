from x2robot_dataset.lazy_dataset import (
    X2RDataProcessingConfig,
    X2RDataChunkConfig,
)

from x2robot_dataset.common.datasets import create_instance
import tqdm

import os, sys
from pathlib import Path

def to_hf_dataset(
        dataset_path,
        data_config,
        force_overwrite=False,
        num_threads=32,
        root_dir=Path('/x2robot/Data/.cache/hf_datasets')
    ) -> None:
    create_instance(data_config).from_raw_to_videolazy_format(
            dataset_path=dataset_path,
            force_overwrite=force_overwrite,
            num_threads=num_threads,
            root_dir=root_dir,
            class_type=data_config['class_type']
    )
    return None

def split_data(data_folders, num_tasks):
    chunk_size = (len(data_folders) + num_tasks - 1) // num_tasks
    return [data_folders[i:i + chunk_size] for i in range(0, len(data_folders), chunk_size)]


if __name__ == '__main__':
    # data_folders = [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/droid_dataset/{i}.zarr.zip' for i in range(93)
    # ]
    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/fractal20220817_data/{i}.zarr.zip' for i in range(3)
    # ]

    data_folders = [
        f'/x2robot/Data/x2robot_Open_X_datasets/dobbe_dataset/{i}.zarr.zip' for i in range(6)
    ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/bridge_data_v2_uid/bridge_data_v2/bridge_data_v2_img1/{i}.zarr.zip' for i in range(34)
    # ]
    
    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/fmb_dataset/{i}.zarr.zip' for i in range(2)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/bridge_data_v2_uid/bridge_data_v2/bridge_data_v2_img3/{i}.zarr.zip' for i in range(20)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/bridge_data_v2_uid/bridge_data_v2/bridge_data_v2_img4/{i}.zarr.zip' for i in range(8)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/bc_z/{i}.zarr.zip' for i in range(44)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/utaustin_mutex/{i}.zarr.zip' for i in range(2)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/stanford_kuka_multimodal_dataset_converted_externally_to_rlds/{i}.zarr.zip' for i in range(2)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/taco_play/{i}.zarr.zip' for i in range(4)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/jaco_play/{i}.zarr.zip' for i in range(2)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/berkeley_cable_routing/{i}.zarr.zip' for i in range(2)
    # ]

    # data_folders += [
    #     f'/x2robot/Data/x2robot_Open_X_datasets/furniture_bench_dataset_converted_externally_to_rlds/{i}.zarr.zip' for i in range(6)
    # ]

    # data_folders += [
    #     '/x2robot/Data/x2robot_Open_X_datasets/stanford_hydra_dataset_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/austin_buds_dataset_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/austin_sailor_dataset_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/nyu_rot_dataset_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/austin_sirius_dataset_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/dlr_edan_shared_control_converted_externally_to_rlds/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/berkeley_autolab_ur5/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/berkeley_fanuc_manipulation/0.zarr.zip', \
    #     '/x2robot/Data/x2robot_Open_X_datasets/viola_dataset/0.zarr.zip', \
    # ]

    num_tasks = int(os.getenv("NUM_TASKS", 10))  # 默认任务数量 10
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))  # 当前 Slurm 任务 ID

    data_splits = split_data(data_folders, num_tasks)

    if task_id >= len(data_splits):
        print(f"Task ID {task_id} is out of range. No data to process.")
        sys.exit(0)
    
    data_subset = data_splits[task_id]
    
    data_configs = [X2RDataProcessingConfig() for _ in range(len(data_subset))]
    for i in range(len(data_configs)):
        data_configs[i] = data_configs[i].update(
            class_type = 'zarr'
        )
    

    data_configs = [config.as_dict() for config in data_configs]

    for data_folder, data_config in tqdm.tqdm(zip(data_subset, data_configs), total=len(data_subset), desc='Converting to hf dataset'):
        print('generating hf dataset for', data_folder)
        to_hf_dataset(
            data_folder,
            data_config,
            force_overwrite=False,
            num_threads=60
        )

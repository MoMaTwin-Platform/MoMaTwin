# data preprocessing for x2robot raw data   

import numpy as np
from collections import defaultdict

import imageio
import json
import os


_CAM_FILE_MAPPING = {
    'face_view': 'faceImg.mp4',
    'left_wrist_view': 'leftImg.mp4',
    'right_wrist_view': 'rightImg.mp4'
}


def process_video(video_path, sampling_freq=20):
    reader = imageio.get_reader(video_path, 'ffmpeg')

    fps = reader.get_meta_data()['fps']

    if fps == sampling_freq:
        frame_arrays = np.array([frame for frame in reader])
    else:
        interval = int(round(fps / sampling_freq))

        frames = []

        for i, frame in enumerate(reader):
            if i % interval == 0:
                frames.append(frame)
        frame_arrays = np.array([np.array(frame) for frame in frames])

    reader.close()

    return frame_arrays

def process_videos(file_path, obs_key=None):
    if obs_key is not None:
        cam_list = []
        cam_desc = []
        for key in obs_key:
            if key in _CAM_FILE_MAPPING:
                cam_list.append(os.path.join(file_path, _CAM_FILE_MAPPING[key]))
                cam_desc.append(key)
    else:
        cam_list, cam_desc = [], []
        for name,cam_file in _CAM_FILE_MAPPING.items():
            if not os.path.exists(os.path.join(file_path, cam_file)):
                continue
            else:
                cam_list.append(os.path.join(file_path, cam_file))
                cam_desc.append(name)

    video = {name:process_video(video_path) for name,video_path in zip(cam_desc, cam_list)}

    return video


_ACTION_KEY_FULL_MAPPING = {
    'follow_right_arm_joint_pos': 'follow_right_joint_pos',
    'follow_right_arm_joint_dev': 'follow_right_joint_dev',
    'follow_right_arm_joint_cur': 'follow_right_joint_cur',
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'master_right_arm_joint_pos': 'master_right_joint_pos',
    'master_right_arm_joint_dev': 'master_right_joint_dev',
    'master_right_arm_joint_cur': 'master_right_joint_cur',
    'master_right_ee_cartesian_pos': 'master_right_position',
    'master_right_ee_rotation': 'master_right_rotation',
    'master_right_gripper': 'master_right_gripper',
    'follow_left_arm_joint_pos': 'follow_left_joint_pos',
    'follow_left_arm_joint_dev': 'follow_left_joint_dev',
    'follow_left_arm_joint_cur': 'follow_left_joint_cur',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper',
    'master_left_arm_joint_pos': 'master_left_joint_pos',
    'master_left_arm_joint_dev': 'master_left_joint_dev',
    'master_left_arm_joint_cur': 'master_left_joint_cur',
    'master_left_ee_cartesian_pos': 'master_left_position',
    'master_left_ee_rotation': 'master_left_rotation',
    'master_left_gripper': 'master_left_gripper'
}
_ACTION_KEY_FULL_MAPPING_INV = {v:k for k,v in _ACTION_KEY_FULL_MAPPING.items()}

_ACTION_KEY_EE_MAPPING = {
    'follow_right_ee_cartesian_pos': 'follow_right_position',
    'follow_right_ee_rotation': 'follow_right_rotation',
    'follow_right_gripper': 'follow_right_gripper',
    'follow_left_ee_cartesian_pos': 'follow_left_position',
    'follow_left_ee_rotation': 'follow_left_rotation',
    'follow_left_gripper': 'follow_left_gripper'
}
_ACTION_KEY_EE_MAPPING_INV = {v:k for k,v in _ACTION_KEY_EE_MAPPING.items()}

def process_action(file_path, action_key_mapping=_ACTION_KEY_FULL_MAPPING):
    action_path = os.path.join(file_path, f"{file_path.split('/')[-1]}.json")
    # Open the JSON file
    trajectories = defaultdict(lambda:[])
    with open(action_path, 'r') as file:
        actions = json.load(file)
        for action in actions['data']:
            for key, val in action.items():
                # Check if the key is in the mapping's values
                if key in action_key_mapping.keys():
                    trajectories[action_key_mapping[key]].append(np.array(val))

        trajectories  = {key:np.array(val, dtype=np.float32) for key,val in trajectories.items()}

        return trajectories

TASK_DESC_MAPPING = {
    'sausage': 'cut the sausage with the knife',
    'cucumber': 'cut cucumber with the knife',
    'juice': 'open the juice powder can, and use the spoon to add the powder to the cup',
    'clean': 'use sponge to clean the table',
    'salad': 'add sauce to the bowl containing cucumbers',
    'flower': 'watering the flower with the watering can',
    'water-flowers': 'watering the flowers with the watering can',
    'clothes': 'fold the clothes',
    'water': 'pour water from kettle into the cup containing juice powder and stir with a spoon to make juice',
    'pour-water': 'pour water from kettle into the cup containing juice powder and stir with a spoon to make juice',
    'wastesorting': 'put dry trash in dry trash bin, and put wet trash in wet trash bin',
}
def process_instruction(file_path):
    task_name = file_path.split('/')[-2]
    for key, val in TASK_DESC_MAPPING.items():
        if key in task_name:
            return {
                "text_en":[val]
            }

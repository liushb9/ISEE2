import sys

sys.path.append("./policy/ACT/")

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import pdb
import json
import torch
import clip
from typing import List


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def text2feats(text_inputs: List[str]):
    """Convert text instructions to CLIP text features."""
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    text_tokens = clip.tokenize(text_inputs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_feat = text_features.detach().cpu().numpy()
    return text_feat.astype(np.float32)


def convert_task_name_to_instruction(task_name: str) -> str:
    task_instruction = task_name.replace("_", " ")
    return f"ur5 {task_instruction}"


def pad_action_14_to_16(action_14: np.ndarray) -> np.ndarray:
    """
    Convert 14-dim action to 16-dim action with padding.
    Format: [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)] = 14
    To: [left_arm(6) + padding(1) + left_gripper(1) + right_arm(6) + padding(1) + right_gripper(1)] = 16
    """
    if action_14.shape[-1] == 14:
        left_arm = action_14[..., :6]
        left_gripper = action_14[..., 6:7]
        right_arm = action_14[..., 7:13]
        right_gripper = action_14[..., 13:14]
        # Pad to 16 dim: (6+padding+1, 6+padding+1)
        action_16 = np.concatenate([
            left_arm,
            np.zeros((*action_14.shape[:-1], 1)),  # padding
            left_gripper,
            right_arm,
            np.zeros((*action_14.shape[:-1], 1)),  # padding
            right_gripper
        ], axis=-1)
        return action_16.astype(np.float32)
    elif action_14.shape[-1] == 16:
        return action_14.astype(np.float32)
    else:
        raise ValueError(f"Unexpected action dimension: {action_14.shape[-1]}")


def get_action_mask(action_dim: int) -> np.ndarray:
    """
    Create a mask indicating valid dimensions in 16-dim action space.
    Returns mask of shape (16,) where 1 indicates valid, 0 indicates padding.
    """
    if action_dim == 14:
        # Mask for 14-dim: [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
        mask = np.array([1, 1, 1, 1, 1, 1, 1,  # left_arm(6) + left_gripper(1)
                         0,  # padding
                         1, 1, 1, 1, 1, 1, 1,  # right_arm(6) + right_gripper(1)
                         0])  # padding
        return mask.astype(np.float32)
    elif action_dim == 16:
        # All dimensions are valid for 16-dim
        return np.ones(16, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected action dimension: {action_dim}")


def data_transform(path, episode_num, save_path, task_name=None):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = (load_hdf5(
            os.path.join(path, f"episode{i}.hdf5")))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            if j != left_gripper_all.shape[0] - 1:
                state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint

                state = state.astype(np.float32)
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                # Convert action to 16-dim if needed
                action_16 = pad_action_14_to_16(action.reshape(1, -1)).reshape(-1)
                actions.append(action_16)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        # Generate text_feat for this task
        if task_name is not None:
            instruction = convert_task_name_to_instruction(task_name)
            text_feat = text2feats([instruction])[0]  # Get single feature vector
        else:
            # Default instruction if task_name not provided
            text_feat = text2feats(["default task"])[0]

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            actions_array = np.array(actions)
            # Detect original action dimension
            original_action_dim = actions_array.shape[1] if len(actions_array) > 0 else 14
            
            # Ensure actions are 16-dim
            if original_action_dim == 14:
                actions_array = pad_action_14_to_16(actions_array)
            
            f.create_dataset("action", data=actions_array)
            obs = f.create_group("observations")
            qpos_array = np.array(qpos)
            # Also pad qpos to 16-dim if needed
            if qpos_array.shape[1] == 14:
                qpos_array = pad_action_14_to_16(qpos_array)
            obs.create_dataset("qpos", data=qpos_array)
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            obs.create_dataset("action_mask", data=get_action_mask(original_action_dim))
            obs.create_dataset("text_feat", data=text_feat)
            image = obs.create_group("images")
            # cam_high_enc, len_high = images_encoding(cam_high)
            # cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            # cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=np.stack(cam_high), dtype=np.uint8)
            image.create_dataset("cam_right_wrist", data=np.stack(cam_right_wrist), dtype=np.uint8)
            image.create_dataset("cam_left_wrist", data=np.stack(cam_left_wrist), dtype=np.uint8)

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., adjust_bottle)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument("expert_data_num", type=int)

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    begin = 0
    begin = data_transform(
        os.path.join("../../data_ur5-wsg/", task_name, task_config, 'data'),
        expert_data_num,
        f"processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        task_name=task_name,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./processed_data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)

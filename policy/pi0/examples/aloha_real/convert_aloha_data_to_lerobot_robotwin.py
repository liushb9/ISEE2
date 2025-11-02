"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


def map_to_unified_state_32dim(state_or_action, unified_dim=32):
    """
    Map 14-dim or 16-dim state/action to 32-dim unified state vector.
    
    Mapping rules:
    - 14-dim: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
      -> [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1), padding(18)]
    - 16-dim: [left_arm(6), padding(1), left_gripper(1), right_arm(6), padding(1), right_gripper(1)]
      -> [left_arm(6), padding(1), left_gripper(1), right_arm(6), padding(1), right_gripper(1), padding(16)]
    
    Args:
        state_or_action: Array with shape (..., 14) or (..., 16)
        unified_dim: Target unified dimension (default 32)
    
    Returns:
        unified_state: Array with shape (..., unified_dim)
        mask: Binary mask indicating valid dimensions, shape (..., unified_dim)
    """
    if isinstance(state_or_action, torch.Tensor):
        device = state_or_action.device
        dtype = state_or_action.dtype
        state_np = state_or_action.detach().cpu().numpy()
        is_torch = True
    else:
        state_np = np.asarray(state_or_action)
        is_torch = False
    
    original_shape = state_np.shape
    state_np = state_np.reshape(-1, state_np.shape[-1])
    original_dim = state_np.shape[-1]
    
    # Create unified state vector
    unified_state = np.zeros((state_np.shape[0], unified_dim), dtype=state_np.dtype)
    mask = np.zeros((state_np.shape[0], unified_dim), dtype=np.float32)
    
    if original_dim == 14:
        # 14-dim: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        unified_state[:, :14] = state_np
        mask[:, :14] = 1.0
    elif original_dim == 16:
        # 16-dim: [left_arm(6), padding(1), left_gripper(1), right_arm(6), padding(1), right_gripper(1)]
        unified_state[:, :16] = state_np
        # Mark valid dimensions (all except padding positions at indices 6 and 14)
        mask[:, :16] = 1.0
        mask[:, 6] = 0.0  # padding dimension
        mask[:, 14] = 0.0  # padding dimension
    else:
        raise ValueError(f"Unsupported dimension: {original_dim}. Expected 14 or 16.")
    
    # Reshape back to original shape (with unified_dim instead of original_dim)
    new_shape = original_shape[:-1] + (unified_dim,)
    unified_state = unified_state.reshape(new_shape)
    mask = mask.reshape(new_shape)
    
    if is_torch:
        unified_state = torch.from_numpy(unified_state).to(device=device, dtype=dtype)
        mask = torch.from_numpy(mask).to(device=device, dtype=dtype)
    
    return unified_state, mask


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    # Use 32-dim unified state vector instead of 14-dim
    unified_dim = 32
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"dim_{i}" for i in range(unified_dim)],
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"dim_{i}" for i in range(unified_dim)],
            ],
        },
        "observation.state_mask": {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"mask_{i}" for i in range(unified_dim)],
            ],
        },
        "action_mask": {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"mask_{i}" for i in range(unified_dim)],
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"dim_{i}" for i in range(unified_dim)],
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (unified_dim, ),
            "names": [
                [f"dim_{i}" for i in range(unified_dim)],
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        # add prompt
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            # Handle both string and list types
            if isinstance(instructions, str):
                instruction = instructions
            elif isinstance(instructions, list):
                instruction = np.random.choice(instructions)
            else:
                raise ValueError(f"instructions must be a string or list, got {type(instructions)}")
        # Convert state and action to unified 32-dim format
        state_unified, state_mask = map_to_unified_state_32dim(state)
        action_unified, action_mask = map_to_unified_state_32dim(action)
        
        # Convert to numpy if torch tensors
        if isinstance(state_unified, torch.Tensor):
            state_unified = state_unified.numpy()
            state_mask = state_mask.numpy()
        if isinstance(action_unified, torch.Tensor):
            action_unified = action_unified.numpy()
            action_mask = action_mask.numpy()
        
        for i in range(num_frames):
            frame = {
                "observation.state": state_unified[i],
                "action": action_unified[i],
                "observation.state_mask": state_mask[i],
                "action_mask": action_mask[i],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                velocity_unified, _ = map_to_unified_state_32dim(velocity)
                if isinstance(velocity_unified, torch.Tensor):
                    velocity_unified = velocity_unified.numpy()
                frame["observation.velocity"] = velocity_unified[i]
            if effort is not None:
                effort_unified, _ = map_to_unified_state_32dim(effort)
                if isinstance(effort_unified, torch.Tensor):
                    effort_unified = effort_unified.numpy()
                frame["observation.effort"] = effort_unified[i]
            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)

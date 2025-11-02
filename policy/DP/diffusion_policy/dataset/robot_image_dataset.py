from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.action_utils import pad_action_14_to_16, create_action_mask
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
import torch
import pdb


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        target_action_dim=16,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action", "text_feat"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.target_action_dim = target_action_dim

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        
        # Create identity normalizer for text_feat (does not normalize)
        # text_feat has shape (..., 512), so we need to create an identity normalizer
        # with 512 dimensions, all with scale=1, offset=0
        if "text_feat" in self.replay_buffer:
            text_feat_dim = self.replay_buffer["text_feat"].shape[-1]
            text_feat_scale = torch.ones(text_feat_dim, dtype=torch.float32)
            text_feat_offset = torch.zeros(text_feat_dim, dtype=torch.float32)
            # Create dummy stats (not used since scale=1, offset=0)
            text_feat_stats = {
                "min": torch.zeros(text_feat_dim, dtype=torch.float32),
                "max": torch.ones(text_feat_dim, dtype=torch.float32),
                "mean": torch.zeros(text_feat_dim, dtype=torch.float32),
                "std": torch.ones(text_feat_dim, dtype=torch.float32),
            }
            text_feat_normalizer = SingleFieldLinearNormalizer.create_manual(
                scale=text_feat_scale,
                offset=text_feat_offset,
                input_stats_dict=text_feat_stats,
            )
            normalizer["text_feat"] = text_feat_normalizer
        
        # Note: text_feat should NOT be normalized - it's already in CLIP embedding space
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        # front_cam = np.moveaxis(sample['front_camera'],-1,1)/255
        # left_cam = np.moveaxis(sample['left_camera'],-1,1)/255
        # right_cam = np.moveaxis(sample['right_camera'],-1,1)/255
        text_feat = sample["text_feat"].astype(np.float32)  # T, D
        
        # Handle action padding: convert 14-dim to 16-dim if needed
        action = sample["action"].astype(np.float32)  # T, D
        original_action_dim = action.shape[-1]
        
        if original_action_dim == 14 and self.target_action_dim == 16:
            action, action_mask = pad_action_14_to_16(action)
        else:
            action_mask = create_action_mask(self.target_action_dim, valid_dim=original_action_dim)

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                # 'front_cam': front_cam, # T, 3, H, W
                # 'left_cam': left_cam, # T, 3, H, W
                # 'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
                "text_feat": text_feat,  # T, D
            },
            "action": action,  # T, D (possibly padded)
            "action_mask": action_mask,  # T, D (mask for valid dimensions)
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        # front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        # left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        # right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        text_feat = samples["text_feat"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        
        # Handle action_mask if it exists
        action_mask = None
        if "action_mask" in samples:
            action_mask = samples["action_mask"].to(device, non_blocking=True)
        
        result = {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                # 'front_cam': front_cam, # B, T, 3, H, W
                # 'left_cam': left_cam, # B, T, 3, H, W
                # 'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
                "text_feat": text_feat,  # B, T, D
            },
            "action": action,  # B, T, D
        }
        
        if action_mask is not None:
            result["action_mask"] = action_mask  # B, T, D
        
        return result


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)

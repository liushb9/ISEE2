import numpy as np
import torch


def pad_action_14_to_16(action_14):
    """
    Pad 14-dim action to 16-dim with zeros.
    
    Structure:
    - 14-dim: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    - 16-dim: [left_arm(6), padding(1), left_gripper(1), right_arm(6), padding(1), right_gripper(1)]
    
    Args:
        action_14: Action with shape (..., 14)
    
    Returns:
        action_16: Action with shape (..., 16)
        mask: Binary mask indicating valid dimensions, shape (..., 16)
    """
    if isinstance(action_14, torch.Tensor):
        device = action_14.device
        dtype = action_14.dtype
        shape = action_14.shape
        action_14_np = action_14.detach().cpu().numpy()
    else:
        action_14_np = action_14
        shape = action_14_np.shape
    
    # Reshape to ensure last dimension is 14
    original_shape = action_14_np.shape
    action_14_np = action_14_np.reshape(-1, 14)
    
    # Split 14-dim into components
    left_arm = action_14_np[:, :6]  # 6 dims
    left_gripper = action_14_np[:, 6:7]  # 1 dim
    right_arm = action_14_np[:, 7:13]  # 6 dims
    right_gripper = action_14_np[:, 13:14]  # 1 dim
    
    # Construct 16-dim action with padding
    action_16_np = np.zeros((action_14_np.shape[0], 16))
    action_16_np[:, :6] = left_arm  # left_arm
    action_16_np[:, 6] = 0  # padding
    action_16_np[:, 7] = left_gripper[:, 0]  # left_gripper
    action_16_np[:, 8:14] = right_arm  # right_arm
    action_16_np[:, 14] = 0  # padding
    action_16_np[:, 15] = right_gripper[:, 0]  # right_gripper
    
    # Create mask: 1 for valid dimensions, 0 for padding
    mask = np.ones((action_14_np.shape[0], 16))
    mask[:, 6] = 0  # padding dimension
    mask[:, 14] = 0  # padding dimension
    
    # Reshape back to original shape (with 16 instead of 14)
    new_shape = original_shape[:-1] + (16,)
    action_16_np = action_16_np.reshape(new_shape)
    mask = mask.reshape(new_shape)
    
    if isinstance(action_14, torch.Tensor):
        action_16 = torch.from_numpy(action_16_np).to(device=device, dtype=dtype)
        mask = torch.from_numpy(mask).to(device=device, dtype=dtype)
        return action_16, mask
    else:
        return action_16_np, mask


def unpad_action_16_to_14(action_16, original_dim=14):
    """
    Convert 16-dim action back to 14-dim or keep as 16-dim.
    
    Args:
        action_16: Action with shape (..., 16)
        original_dim: Original action dimension (14 or 16)
    
    Returns:
        action: Action with shape (..., original_dim)
    """
    if original_dim == 16:
        return action_16
    
    if isinstance(action_16, torch.Tensor):
        device = action_16.device
        dtype = action_16.dtype
        shape = action_16.shape
        action_16_np = action_16.detach().cpu().numpy()
    else:
        action_16_np = action_16
        shape = action_16_np.shape
    
    # Reshape to ensure last dimension is 16
    original_shape = action_16_np.shape
    action_16_np = action_16_np.reshape(-1, 16)
    
    # Extract components
    left_arm = action_16_np[:, :6]  # 6 dims
    left_gripper = action_16_np[:, 7:8]  # 1 dim (skip padding at index 6)
    right_arm = action_16_np[:, 8:14]  # 6 dims
    right_gripper = action_16_np[:, 15:16]  # 1 dim (skip padding at index 14)
    
    # Concatenate to 14-dim
    action_14_np = np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1)
    
    # Reshape back to original shape (with 14 instead of 16)
    new_shape = original_shape[:-1] + (14,)
    action_14_np = action_14_np.reshape(new_shape)
    
    if isinstance(action_16, torch.Tensor):
        action_14 = torch.from_numpy(action_14_np).to(device=device, dtype=dtype)
        return action_14
    else:
        return action_14_np


def create_action_mask(action_dim, valid_dim=14):
    """
    Create a mask for action space.
    
    Args:
        action_dim: Current action dimension (14 or 16)
        valid_dim: Original valid dimension (14)
    
    Returns:
        mask: Binary mask, shape (action_dim,)
    """
    if action_dim == 14:
        return np.ones(14)
    elif action_dim == 16:
        mask = np.ones(16)
        mask[6] = 0  # padding dimension
        mask[14] = 0  # padding dimension
        return mask
    else:
        raise ValueError(f"Unsupported action_dim: {action_dim}")


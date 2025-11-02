#!/usr/bin/env python3
"""
Merge all zarr datasets from data/ directory into a single integrated dataset.
Converts 14-dim actions to 16-dim with padding: 6+padding+1+6+padding+1
"""

import os
import zarr
import numpy as np
import sys
from pathlib import Path

# Add the diffusion_policy directory to path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_dir, 'diffusion_policy'))
# Note: We define pad_action_14_to_16_simple locally instead of importing


def pad_action_14_to_16_simple(action_14):
    """
    Simple version that only pads, no mask.
    Structure: 6+padding+1+6+padding+1
    """
    action_14_np = np.asarray(action_14)
    original_shape = action_14_np.shape
    
    # Reshape to ensure last dimension is 14
    action_14_np = action_14_np.reshape(-1, 14)
    
    # Split 14-dim into components
    left_arm = action_14_np[:, :6]  # 6 dims
    left_gripper = action_14_np[:, 6:7]  # 1 dim
    right_arm = action_14_np[:, 7:13]  # 6 dims
    right_gripper = action_14_np[:, 13:14]  # 1 dim
    
    # Construct 16-dim action with padding: 6+padding+1+6+padding+1
    action_16_np = np.zeros((action_14_np.shape[0], 16), dtype=action_14_np.dtype)
    action_16_np[:, :6] = left_arm  # left_arm (6)
    action_16_np[:, 6] = 0  # padding
    action_16_np[:, 7] = left_gripper[:, 0]  # left_gripper (1)
    action_16_np[:, 8:14] = right_arm  # right_arm (6)
    action_16_np[:, 14] = 0  # padding
    action_16_np[:, 15] = right_gripper[:, 0]  # right_gripper (1)
    
    # Reshape back to original shape (with 16 instead of 14)
    new_shape = original_shape[:-1] + (16,)
    action_16_np = action_16_np.reshape(new_shape)
    
    return action_16_np


def load_zarr_data(zarr_path):
    """Load data from a zarr file."""
    print(f"Loading: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    data = {}
    required_keys = ['head_camera', 'state', 'action', 'text_feat']
    
    for key in required_keys:
        if key in root['data']:
            data[key] = root['data'][key][:]
        else:
            raise KeyError(f"Required key '{key}' not found in {zarr_path}")
    
    # Load episode_ends
    if 'episode_ends' not in root['meta']:
        raise KeyError(f"episode_ends not found in meta of {zarr_path}")
    episode_ends = root['meta']['episode_ends'][:]
    
    return data, episode_ends


def merge_zarr_datasets(data_dir, output_path):
    """
    Merge all zarr datasets in data_dir into a single dataset.
    
    Args:
        data_dir: Directory containing zarr files
        output_path: Output zarr file path
    """
    # Find all zarr files
    zarr_files = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.endswith('.zarr'):
            zarr_files.append(item_path)
    
    zarr_files.sort()  # Sort for consistent ordering
    print(f"Found {len(zarr_files)} zarr files to merge")
    
    # Initialize lists for merged data
    all_head_camera = []
    all_state = []
    all_action = []
    all_text_feat = []
    all_episode_ends = []
    
    total_frames = 0
    ref_shapes = {}  # Reference shapes for validation
    
    # Process each zarr file
    for idx, zarr_file in enumerate(zarr_files):
        print(f"\n[{idx+1}/{len(zarr_files)}] Processing: {os.path.basename(zarr_file)}")
        
        try:
            data, episode_ends = load_zarr_data(zarr_file)
            
            # Process action: pad 14-dim to 16-dim if needed
            action = data['action']
            action_dim = action.shape[-1]
            
            if action_dim == 14:
                print(f"  Padding action from 14-dim to 16-dim")
                action = pad_action_14_to_16_simple(action)
            elif action_dim == 16:
                print(f"  Action already 16-dim, keeping as is")
            else:
                print(f"  Warning: Unexpected action dimension {action_dim}, keeping as is")
            data['action'] = action
            
            # Process state: pad 14-dim to 16-dim if needed (same structure as action)
            state = data['state']
            state_dim = state.shape[-1]
            
            if state_dim == 14:
                print(f"  Padding state from 14-dim to 16-dim")
                state = pad_action_14_to_16_simple(state)
            elif state_dim == 16:
                print(f"  State already 16-dim, keeping as is")
            else:
                print(f"  Warning: Unexpected state dimension {state_dim}, keeping as is")
            data['state'] = state
            
            # Check shapes (except first dimension) - use first file as reference
            if idx == 0:
                for key in ['head_camera', 'state', 'action', 'text_feat']:
                    ref_shapes[key] = data[key].shape[1:]
            else:
                for key in ['head_camera', 'state', 'action', 'text_feat']:
                    if data[key].shape[1:] != ref_shapes[key]:
                        print(f"  Warning: Shape mismatch for {key}: {data[key].shape[1:]} vs {ref_shapes[key]}")
            
            # Append data
            all_head_camera.append(data['head_camera'])
            all_state.append(data['state'])
            all_action.append(data['action'])
            all_text_feat.append(data['text_feat'])
            
            # Update episode_ends (cumulative)
            new_episode_ends = episode_ends + total_frames
            all_episode_ends.append(new_episode_ends)
            
            # Update total frames
            total_frames += len(data['head_camera'])
            
            print(f"  Episodes: {len(episode_ends)}, Frames: {len(data['head_camera'])}, Total frames so far: {total_frames}")
            
        except Exception as e:
            print(f"  Error processing {zarr_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_head_camera) == 0:
        print("Error: No data was successfully loaded!")
        return
    
    # Concatenate all data
    print(f"\nConcatenating all data...")
    merged_head_camera = np.concatenate(all_head_camera, axis=0)
    merged_state = np.concatenate(all_state, axis=0)
    merged_action = np.concatenate(all_action, axis=0)
    merged_text_feat = np.concatenate(all_text_feat, axis=0)
    merged_episode_ends = np.concatenate(all_episode_ends)
    
    print(f"Merged data shapes:")
    print(f"  head_camera: {merged_head_camera.shape}")
    print(f"  state: {merged_state.shape}")
    print(f"  action: {merged_action.shape}")
    print(f"  text_feat: {merged_text_feat.shape}")
    print(f"  episode_ends: {merged_episode_ends.shape} (total episodes: {len(merged_episode_ends)})")
    
    # Create output zarr file
    print(f"\nSaving to: {output_path}")
    if os.path.exists(output_path):
        import shutil
        print(f"Removing existing file: {output_path}")
        shutil.rmtree(output_path)
    
    zarr_root = zarr.group(output_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Save data
    chunk_size = (100, *merged_head_camera.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=merged_head_camera,
        chunks=chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    
    chunk_size = (100, merged_state.shape[1])
    zarr_data.create_dataset(
        "state",
        data=merged_state,
        chunks=chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    
    chunk_size = (100, merged_action.shape[1])
    zarr_data.create_dataset(
        "action",
        data=merged_action,
        chunks=chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    
    chunk_size = (100, merged_text_feat.shape[1])
    zarr_data.create_dataset(
        "text_feat",
        data=merged_text_feat,
        chunks=chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    
    zarr_meta.create_dataset(
        "episode_ends",
        data=merged_episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    
    print(f"\n✅ Successfully merged {len(zarr_files)} zarr files into {output_path}")
    print(f"   Total episodes: {len(merged_episode_ends)}")
    print(f"   Total frames: {len(merged_head_camera)}")


def main():
    # Default paths
    data_dir = "./data"
    output_path = "./data/six_tasks-integrated_clean-1200.zarr"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    merge_zarr_datasets(data_dir, output_path)


if __name__ == "__main__":
    main()


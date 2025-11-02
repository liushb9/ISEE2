# Summary of Changes: Text Features and Action Space Handling

This document summarizes the changes made to support CLIP text encoding and action space handling for both 14-dim and 16-dim data.

## 1. CLIP Text Encoder for Language Instructions

### New Files Created:
- `policy/DP/diffusion_policy/common/text_encoder.py`: Contains CLIP text encoder utilities
  - `text2feats()`: Converts text inputs to CLIP embeddings using RN50
  - `get_task_instruction()`: Converts task names (e.g., "blocks_ranking_rgb") to instructions ("blocks ranking rgb")

### Modified Files:

#### 1.1 `policy/DP/process_data.py`
- Added import for text encoder utilities
- Generates text features using CLIP RN50 encoder for each task
- Stores text features in zarr dataset as `text_feat` key
- Text features have shape (512,) from CLIP RN50

#### 1.2 `policy/DP/diffusion_policy/dataset/robot_image_dataset.py`
- Added `text_feat` to keys loaded from zarr
- Includes `text_feat` in observation dictionary
- Comments note that text_feat should NOT be normalized (already in CLIP embedding space)
- Passes text_feat through dataset pipeline

#### 1.3 `policy/DP/diffusion_policy/model/vision/multi_image_obs_encoder.py`
- Added support for `text` type observations
- Creates MLP for each text feature key that projects from 512-dim to agent_pos dimension
- MLP structure: Linear(512 → agent_pos_dim*2) → ReLU → Linear(agent_pos_dim*2 → agent_pos_dim)
- Processes text features through MLP before concatenating with other features

#### 1.4 `policy/DP/diffusion_policy/config/task/default_task_14.yaml` and `default_task_16.yaml`
- Added `text_feat` to shape_meta:
  ```yaml
  text_feat:
    shape: [512]
    type: text
  ```

#### 1.5 `policy/DP/diffusion_policy/env_runner/dp_runner.py`
- Added conditional inclusion of `text_feat` in observation dictionary during action prediction

#### 1.6 `policy/DP/deploy_policy.py`
- Modified `encode_obs()` to optionally include text features when task_name is provided
- Modified `eval()` to accept `task_name` parameter for generating text features
- Generates text features using CLIP encoder during evaluation

## 2. Action Space Handling (14-dim to 16-dim)

### New Files Created:
- `policy/DP/diffusion_policy/common/action_utils.py`: Contains action space utilities
  - `pad_action_14_to_16()`: Converts 14-dim actions to 16-dim with padding
  - `unpad_action_16_to_14()`: Converts 16-dim actions back to 14-dim
  - `create_action_mask()`: Creates binary mask for valid action dimensions

### Action Structure:
- **14-dim**: `[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`
- **16-dim**: `[left_arm(6), padding(1), left_gripper(1), right_arm(6), padding(1), right_gripper(1)]`
- Valid dimensions marked with mask: padding positions (indices 6 and 14) have mask=0

### Modified Files:

#### 2.1 `policy/DP/diffusion_policy/dataset/robot_image_dataset.py`
- Added `target_action_dim` parameter to constructor (default=16)
- In `_sample_to_data()`:
  - Pads 14-dim actions to 16-dim if needed
  - Creates action mask for valid dimensions
  - Returns both padded action and mask
- In `postprocess()`:
  - Passes through action_mask if it exists

#### 2.2 `policy/DP/deploy_policy.py`
- Modified `eval()` to accept `original_action_dim` parameter
- Unpads 16-dim actions back to 14-dim before executing

## Usage Examples

### Processing Data with Text Features:
```bash
python policy/DP/process_data.py blocks_ranking_rgb aloha_clean 50
```

Or using the shell script:
```bash
cd policy/DP
bash process_data.sh blocks_ranking_rgb aloha_clean 50
```

**Note:** The script reads from `data_aloha-agilex/` directory and generates instructions with "aloha " prefix (e.g., "aloha blocks ranking rgb").

### Training with Text Features:
The dataset will automatically:
1. Load text_feat from zarr
2. Pass it through the vision encoder's MLP
3. Concatenate with other features

### Evaluation with Text Features:
```python
from policy.DP.deploy_policy import get_model, eval, encode_obs

model = get_model(usr_args)
obs = encode_obs(observation, task_name="blocks_ranking_rgb")
eval(TASK_ENV, model, observation, task_name="blocks_ranking_rgb", original_action_dim=14)
```

## Key Design Decisions:

1. **Text Features**: Fixed per task (not per frame) and stored in zarr for each observation frame
2. **No Normalization**: Text features from CLIP are NOT normalized as they're already in a learned embedding space
3. **MLP Processing**: Text features are processed through an MLP to match the dimensionality of agent_pos
4. **Action Padding**: 14-dim actions are padded to 16-dim during training but can be unpadded during evaluation
5. **Backward Compatibility**: All changes maintain backward compatibility when text_feat is not present

## Dependencies:
- `clip` package (for CLIP text encoder)
- `torch` (for neural network operations)
- `numpy` (for array operations)


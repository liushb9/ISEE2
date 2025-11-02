import numpy as np
from .dp_model import DP
import yaml
import sys
import os

# Add the diffusion_policy directory to path
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_dir, 'diffusion_policy'))
from common.text_encoder import text2feats, get_task_instruction
from common.action_utils import unpad_action_16_to_14

def encode_obs(observation, task_name=None):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    
    # Add text feature if task_name is provided
    if task_name is not None:
        instruction = get_task_instruction(task_name)
        text_feat = text2feats([instruction])  # Shape: (1, 512)
        text_feat = text_feat.squeeze(0)  # Shape: (512,)
        obs["text_feat"] = text_feat
    
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']
    
    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation, task_name=None, original_action_dim=14):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    task_name: Task name for generating text features
    original_action_dim: Original action dimension (14 or 16)
    """
    obs = encode_obs(observation, task_name=task_name)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)
    
    # Unpad actions if needed (convert 16-dim back to 14-dim)
    if original_action_dim == 14 and actions.shape[-1] == 16:
        actions = unpad_action_16_to_14(actions, original_dim=14)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, task_name=task_name)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()

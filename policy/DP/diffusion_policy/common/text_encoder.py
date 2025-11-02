import torch
import clip
import numpy as np
from typing import List


def text2feats(text_inputs: List[str]):
    """
    Convert text inputs to CLIP embeddings.
    
    Args:
        text_inputs: List of text strings (e.g., ["blocks ranking rgb"])
    
    Returns:
        text_feat: numpy array of text features, shape (N, 512) for RN50
    """
    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    
    # Tokenize and encode
    text_tokens = clip.tokenize(text_inputs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    
    # Convert to numpy
    text_feat = text_features.detach().cpu().numpy()
    return text_feat.astype(np.float32)


def get_task_instruction(task_name: str) -> str:
    """
    Convert task name to instruction.
    Example: 'blocks_ranking_rgb' -> 'blocks ranking rgb'
    
    Args:
        task_name: Task name with underscores
    
    Returns:
        instruction: Human-readable instruction
    """
    # Replace underscores with spaces
    instruction = task_name.replace('_', ' ')
    return instruction


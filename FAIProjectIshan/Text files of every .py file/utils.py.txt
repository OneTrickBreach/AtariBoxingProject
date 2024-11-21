# training/utils.py

import torch

def save_checkpoint(model_state_dict, filename):
    """
    Save model checkpoint.

    Args:
        model_state_dict (dict): State dictionary of the model to save.
        filename (str): File path to save the checkpoint.
    """
    torch.save(model_state_dict, filename)
    
def load_checkpoint(filename):
    """
    Load model checkpoint.

    Args:
        filename (str): File path of the checkpoint to load.

    Returns:
         dict: Loaded state dictionary of the model.
    """
    return torch.load(filename)
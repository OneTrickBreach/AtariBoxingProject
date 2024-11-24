# training/utils.py
import os
import torch

def save_checkpoint(model_state_dict, filename):
    """
    Save a model checkpoint to the specified filename.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the checkpoint
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
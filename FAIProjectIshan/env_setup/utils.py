# env_setup/utils.py
import numpy as np
import random
import torch

def set_seed(seed):
    """
    Sets random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_message(message, log_file=None):
    """
    Logs a message to the console and optionally to a file.

    Args:
        message (str): Message to log.
        log_file (str, optional): File to write logs. Defaults to None.
    """
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")

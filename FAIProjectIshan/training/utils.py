# training/utils.py
import os

def save_config(config, file_path):
    """
    Saves a configuration dictionary to a file.

    Args:
        config (dict): Configuration dictionary.
        file_path (str): Path to save the configuration.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def load_config(file_path):
    """
    Loads a configuration dictionary from a file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            config[key] = eval(value)  # Use eval to interpret Python literals
    return config

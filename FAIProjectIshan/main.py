import argparse
import yaml
from training.train import train
from training.evaluation import evaluate
from env_setup.env_wrapper import EnvironmentWrapper
from models.cnn_model import CNNModel
import os

def load_config(config_path):
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent RL using PettingZoo Atari Boxing")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], required=True, help="Mode: train or evaluate")
    parser.add_argument('--save_path', type=str, default="models/", help="Path to save trained models")
    parser.add_argument('--load_path', type=str, default="models/", help="Path to load trained models (for evaluation)")
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)
    env_config = config['env']
    agent_config = config['agent']
    training_config = config['training']
    train(env_config, agent_config, training_config, save_path=args.save_path)

    # Check the specified mode
    if args.mode == 'train':
        print("Starting training...")
        train(env_config, agent_config, training_config, save_path=args.save_path)
    elif args.mode == 'evaluate':
        print("Starting evaluation...")
        evaluate(env_config, agent_config, load_path=args.load_path, num_episodes=training_config.get('evaluation_episodes', 10))
    else:
        raise ValueError("Invalid mode selected. Choose either 'train' or 'evaluate'.")

if __name__ == "__main__":
    main()

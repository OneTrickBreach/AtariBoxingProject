import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
from agents.multi_agent import MultiAgentRL
from training.replay_buffer import ReplayBuffer
from training.utils import save_checkpoint

# Preprocessing pipeline for observations
preprocess = Compose([
    ToTensor(),               # Convert to tensor
    Resize((84, 84)),         # Resize to [84, 84]
    Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])


def preprocess_observation(obs):
    """
    Preprocess a single observation (resize, normalize, and permute dimensions).
    """
    # If input is already a tensor, just normalize it
    if isinstance(obs, torch.Tensor):
        if obs.max() > 1:  # If the tensor values are not normalized
            obs = obs / 255.0  # Normalize from [0, 255] to [0, 1]
    else:
        # If it's a PIL image or numpy array, convert it to tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32) / 255.0
        elif isinstance(obs, PIL.Image.Image):
            # Apply the transforms only if it's a PIL image
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((84, 84)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            obs = preprocess(obs)  # Apply preprocessing pipeline

    # Ensure the shape is in the right format (C, H, W)
    return obs.permute(2, 0, 1) if len(obs.shape) == 3 else obs  # For image channels, handle as needed
 # Ensure correct shape (C, H, W)
  # Ensure correct shape (C, H, W)

def train(num_episodes=1000, batch_size=32, target_update_freq=10, gamma=0.99):
    """
    Main training loop for multi-agent reinforcement learning.
    """
    multi_agent = MultiAgentRL()  # Initialize the multi-agent environment
    buffer = ReplayBuffer(capacity=10000)  # Initialize replay buffer

    for episode in range(num_episodes):
        obs_tuple = multi_agent.reset_environment()  # Reset the environment
        obs_dict = obs_tuple[0]  # Get the initial observations

        # Initialize observation stacks with the first observation (repeated 4 times)
        obs_stack_1 = torch.cat([preprocess_observation(obs_dict["first_0"]).unsqueeze(0)] * 4, dim=0)
        obs_stack_2 = torch.cat([preprocess_observation(obs_dict["second_0"]).unsqueeze(0)] * 4, dim=0)


        # Flags to track termination status
        done_flags = {"first_0": False, "second_0": False}
        episode_reward = {"first_0": 0, "second_0": 0}

        while not all(done_flags.values()):
            # Select actions based on the current observation stacks
            action_1 = multi_agent.agent_1.select_action(obs_stack_1, step=episode)
            action_2 = multi_agent.agent_2.select_action(obs_stack_2, step=episode)
            actions = {"first_0": action_1, "second_0": action_2}

            # Environment step
            next_obs_tuple = multi_agent.step(actions)
            next_obs_dict, rewards_dict, terminations, truncations, infos = next_obs_tuple

            # Compute next observation stacks (keeping the last 3 frames and adding the new one)
            next_obs_stack_1 = torch.cat([obs_stack_1[1:], preprocess_observation(next_obs_dict["first_0"]).unsqueeze(0)], dim=0)
            next_obs_stack_2 = torch.cat([obs_stack_2[1:], preprocess_observation(next_obs_dict["second_0"]).unsqueeze(0)], dim=0)

            # Add transitions to replay buffer
            buffer.add(obs_stack_1.clone(), action_1, rewards_dict["first_0"], next_obs_stack_1.clone(), terminations["first_0"])
            buffer.add(obs_stack_2.clone(), action_2, rewards_dict["second_0"], next_obs_stack_2.clone(), terminations["second_0"])

            # Update episode rewards
            episode_reward["first_0"] += rewards_dict["first_0"]
            episode_reward["second_0"] += rewards_dict["second_0"]

            # Update stacks and done flags for the next iteration
            obs_stack_1 = next_obs_stack_1
            obs_stack_2 = next_obs_stack_2
            done_flags = terminations

            # Sample a batch and update agents if enough transitions exist in the buffer
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                multi_agent.agent_1.update(batch)
                multi_agent.agent_2.update(batch)

        # Update target networks periodically
        if episode % target_update_freq == 0:
            multi_agent.agent_1.update_target_network()
            multi_agent.agent_2.update_target_network()

        # Log episode rewards
        print(f"Episode {episode + 1}/{num_episodes} - Reward Agent 1: {episode_reward['first_0']}, Reward Agent 2: {episode_reward['second_0']}")

        # Save model checkpoints periodically
        if episode % 50 == 0:
            save_checkpoint(multi_agent.agent_1.q_network.state_dict(), f"checkpoints/agent1_episode_{episode}.pth")
            save_checkpoint(multi_agent.agent_2.q_network.state_dict(), f"checkpoints/agent2_episode_{episode}.pth")

if __name__ == "__main__":
    train(num_episodes=10000)

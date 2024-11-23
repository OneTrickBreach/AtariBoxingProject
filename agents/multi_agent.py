from pettingzoo.atari import boxing_v2
from agents.agent import DQNAgent
import numpy as np

class MultiAgentRL:
    def __init__(self):
        """
        Initialize two DQN agents for multi-agent reinforcement learning in PettingZoo's Boxing environment.
        
         - Agent 1 plays as Player 1 in Boxing.
         - Agent 2 plays as Player 2 in Boxing.
         """
        
        # Create environment instance from PettingZoo's Boxing environment
        self.env = boxing_v2.parallel_env()
         
        # Number of possible actions in Boxing environment (6 discrete actions per player)
        num_actions = self.env.action_space('first_0').n

        # Initialize two DQN agents with shared architecture but separate policies
        input_channels = 4  # Assuming we're stacking 4 frames as input

        self.agent_1 = DQNAgent(input_channels=input_channels, num_actions=num_actions)
        self.agent_2 = DQNAgent(input_channels=input_channels, num_actions=num_actions)

        # Initialize step counter
        self.step_counter = 0

    def reset_environment(self):
        """
        Reset the environment and return initial observations for both agents.
        """
        return self.env.reset()

    def step(self, actions):
        """
        Take a step in the environment with both agents' actions.

        Args:
            actions (dict): Dictionary containing actions for both agents.

        Returns:
            dict: Observations after taking a step in the environment.
            dict: Rewards received by both agents after taking their respective actions.
            dict: Done flags indicating whether each agent has finished its episode.
            dict: Additional info from environment.
        """
        return self.env.step(actions)

    def train_agents(self, num_episodes=1000, batch_size=32, target_update_freq=100):
        """
        Main loop for training both agents in parallel mode. This can include logic such as replay buffer,
        updating networks periodically, etc., but is left simple here for clarity.
        """

        obs_dict = self.reset_environment()

        done_flags = {"first_0": False, "second_0": False}

        # Training loop
        for episode in range(num_episodes):
            while not all(done_flags.values()):
                # Get actions from both agents based on their respective observations
                action_1 = self.agent_1.select_action(obs_dict["first_0"], step=self.step_counter)  # Use the step counter for epsilon decay
                action_2 = self.agent_2.select_action(obs_dict["second_0"], step=self.step_counter)

                actions = {"first_0": action_1, "second_0": action_2}

                obs_dict_next, rewards_dict, done_flags, info_dict = self.step(actions)

                # Store transitions in replay buffer and call `update()` on both agents
                self.agent_1.update((obs_dict["first_0"], action_1, rewards_dict["first_0"], obs_dict_next["first_0"], done_flags["first_0"]))
                self.agent_2.update((obs_dict["second_0"], action_2, rewards_dict["second_0"], obs_dict_next["second_0"], done_flags["second_0"]))

                # Update the observation dictionary
                obs_dict = obs_dict_next

                # Update the target networks periodically
                if self.step_counter % target_update_freq == 0:
                    self.agent_1.update_target_network()
                    self.agent_2.update_target_network()

                self.step_counter += 1

            # Optionally print episode info here
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} completed.")


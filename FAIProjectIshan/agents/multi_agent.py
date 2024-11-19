# agents/multi_agent.py
from agents.agent import DQNAgent
import numpy as np

class MultiAgent:
    def __init__(self, num_agents, state_shape, action_size, model_class, **agent_kwargs):
        """
        Initializes the Multi-Agent system.

        Args:
            num_agents (int): Number of agents.
            state_shape (tuple): Shape of the input state.
            action_size (int): Number of possible actions.
            model_class (torch.nn.Module): Class for the CNN model.
            agent_kwargs: Additional arguments for initializing individual agents.
        """
        self.agents = [
            DQNAgent(state_shape, action_size, model_class(*state_shape, action_size), **agent_kwargs)
            for _ in range(num_agents)
        ]

    def act(self, states):
        """
        Selects actions for all agents.

        Args:
            states (list): List of states for each agent.

        Returns:
            list: List of actions for each agent.
        """
        return [agent.act(state) for agent, state in zip(self.agents, states)]

    def store_experience(self, states, actions, rewards, next_states, dones):
        """
        Stores experiences for all agents.

        Args:
            states, actions, rewards, next_states, dones: Experience tuples for all agents.
        """
        for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
            agent.store_experience(state, action, reward, next_state, done)

    def replay(self):
        """
        Triggers experience replay for all agents.
        """
        for agent in self.agents:
            agent.replay()

    def update_target_models(self):
        """
        Updates the target models for all agents.
        """
        for agent in self.agents:
            agent.update_target_model()

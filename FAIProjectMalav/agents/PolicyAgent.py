# agents/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .base import BaseAgent

class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(np.prod(observation_shape), 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_space.n)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_value = self.value_head(x)
        return action_logits, state_value

class PPOAgent(BaseAgent):
    def __init__(self, agent_id, observation_shape, action_space, lr=0.001, gamma=0.99, clip_epsilon=0.2):
        super().__init__(agent_id, action_space)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.policy_network = PolicyNetwork(observation_shape, action_space)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def select_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action_logits, _ = self.policy_network(observation)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        return action.item()

    def train(self, observations, actions, rewards, old_log_probs, advantages):
        
        pass

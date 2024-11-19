# agents/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNAgent:
    def __init__(self, state_shape, action_size, model, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        """
        Initializes a DQN Agent.

        Args:
            state_shape (tuple): Shape of the input state (C, H, W).
            action_size (int): Number of possible actions.
            model (torch.nn.Module): CNN model for value estimation.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay factor for epsilon.
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.model = model
        self.target_model = type(model)(*state_shape, action_size).to(device)  # Target network
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=100000)
        self.batch_size = 64

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            int: Selected action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores an experience in the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state after the action.
            done: Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Samples a batch of experiences and trains the model.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute current Q values
        current_q = self.model(states).gather(1, actions)

        # Compute target Q values
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """
        Updates the target model to match the current model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

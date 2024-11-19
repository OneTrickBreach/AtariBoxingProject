import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent:
    def __init__(self, observation_shape, action_space, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.observation_shape = observation_shape
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        

        input_dim = np.prod(self.observation_shape) 
        print(f"Expected input dimension: {input_dim}")  
        
        self.q_network = self.build_network(input_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = []
        self.batch_size = 64

    def build_network(self, input_dim):
    
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n)
        )

    def select_action(self, observation):
        if random.random() < self.epsilon:
            return self.action_space.sample() 
        else:
            with torch.no_grad():
                observation_tensor = torch.tensor(observation, dtype=torch.float32)
                q_values = self.q_network(observation_tensor)
                return torch.argmax(q_values).item()

    def train_step(self, observation, reward, done, next_observation):

        self.memory.append((observation, reward, done, next_observation))

        if len(self.memory) < self.batch_size:
            return

    
        batch = random.sample(self.memory, self.batch_size)
        states, rewards, dones, next_states = zip(*batch)


        states_image = np.array([state[0] for state in states])
        next_states_image = np.array([next_state[0] for next_state in next_states])

        states_image = torch.tensor(states_image, dtype=torch.float32)
        next_states_image = torch.tensor(next_states_image, dtype=torch.float32)

        states_image = states_image.view(states_image.size(0), -1) 
        next_states_image = next_states_image.view(next_states_image.size(0), -1)


        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.q_network(states_image)
        next_q_values = self.q_network(next_states_image)
        max_next_q_values = next_q_values.max(dim=1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

  
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

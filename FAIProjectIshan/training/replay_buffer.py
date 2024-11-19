# training/replay_buffer.py
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=64):
        """
        Initializes a replay buffer for experience replay.

        Args:
            buffer_size (int): Maximum number of experiences to store.
            batch_size (int): Number of experiences to sample during training.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def store_experience(self, state, action, reward, next_state, done):
        """
        Stores a new experience in the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observed.
            done: Whether the episode ended.
        """
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """
        Samples a batch of experiences from the buffer.

        Returns:
            tuple: Batched states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)

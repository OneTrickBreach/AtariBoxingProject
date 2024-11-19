import gym
import numpy as np

class RewardNormalizer(gym.Wrapper):
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = reward * self.scale
        return observation, reward, done, info

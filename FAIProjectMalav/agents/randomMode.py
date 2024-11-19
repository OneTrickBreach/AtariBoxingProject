from .base import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    def __init__(self, agent_id, action_space):
        super().__init__(agent_id, action_space)

    def select_action(self, observation):
        return self.action_space.sample()

import numpy as np

class BaseAgent:
    def __init__(self, agent_id, action_space):
        self.agent_id = agent_id
        self.action_space = action_space

    def select_action(self, observation):
        raise NotImplementedError("This method should be overridden by subclasses")

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass

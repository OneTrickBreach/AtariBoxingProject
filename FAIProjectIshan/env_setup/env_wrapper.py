from pettingzoo.atari import boxing_v2
from pettingzoo.utils.conversions import parallel_to_aec
import numpy as np
from collections import deque

class FrameStackWrapper:
    def __init__(self, env, num_stack):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_stacked_frames()

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_frames(), reward, termination, truncation, info

    def _get_stacked_frames(self):
        return np.concatenate(list(self.frames), axis=-1)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

class EnvironmentWrapper:
    def __init__(self, frame_stack, resize_to, render_mode, rom_path, state_shape=None):
        self.frame_stack = frame_stack
        self.resize_to = resize_to
        self.render_mode = render_mode
        self.rom_path = rom_path
        self.state_shape = state_shape

        # Initialize the environment
        self.env = boxing_v2.parallel_env(render_mode=self.render_mode, auto_rom_install_path=self.rom_path)
        self.env = parallel_to_aec(self.env)
        self.env = FrameStackWrapper(self.env, num_stack=self.frame_stack)

        # Pass-through the attributes of the original environment
        self.possible_agents = self.env.env.possible_agents
        self.action_space = self.env.env.action_space
        self.observation_space = self.env.env.observation_space
        self.reward_range = self.env.env.reward_range
        self.metadata = self.env.env.metadata
        self.spec = self.env.env.spec

        # Optionally resize observations (implement resizing if needed)
        if self.resize_to:
            print("Resizing wrapper currently not implemented")

    # Ensure methods like reset and step are correctly forwarded
    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, actions):
        return self.env.step(actions)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

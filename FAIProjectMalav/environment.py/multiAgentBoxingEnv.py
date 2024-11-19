from pettingzoo.atari import boxing_v2

class BoxingEnvironment:
    def __init__(self, render_mode="human", seed=None):
        self.render_mode = render_mode
        self.seed = seed
        self.env = None

    def initialize_environment(self):
        self.env = boxing_v2.parallel_env(render_mode=self.render_mode)
        if self.seed is not None:
            self.env.reset(seed=self.seed)
        return self.env

    def reset(self):
        if self.env is None:
            self.initialize_environment()
        observations, infos = self.env.reset()
        return observations, infos

    def close(self):
        if self.env:
            self.env.close()

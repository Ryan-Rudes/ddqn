from gym import Wrapper

class Logger(Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.episode = 0
        self.frame = 0
        self.highscore = 0
        self.score = 0
        self.terminal = False

    def reset(self):
        self.episode += 1
        self.frame += 1
        self.score = 0
        self.terminal = False
        return self.env.reset()

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self.terminal = terminal
        self.frame += 1
        self.score += reward
        self.highscore = max(self.score, self.highscore)
        return observation, reward, terminal, info

from gym import Wrapper
import cv2

class Grayscale(Wrapper):
    def __init__(self, env):
        super(Grayscale, self).__init__(env)

    def preprocess(self, observation):
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    def reset(self):
        observation = self.env.reset()
        return self.preprocess(observation)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.preprocess(observation), reward, terminal, info

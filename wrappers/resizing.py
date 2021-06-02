from gym import Wrapper
import cv2

class Resizing(Wrapper):
    def __init__(self, env, width, height):
        super(Resizing, self).__init__(env)
        self.width = width
        self.height = height

    def resize(self, observation):
        return cv2.resize(observation, (self.width, self.height), interpolation = cv2.INTER_AREA)

    def reset(self):
        observation = self.env.reset()
        return self.resize(observation)

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.resize(observation), reward, terminal, info

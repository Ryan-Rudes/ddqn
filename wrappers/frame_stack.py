from collections import deque
from gym import Wrapper
import numpy as np
import cv2

class FrameStack(Wrapper):
    def __init__(self, env, length=4):
        super(FrameStack, self).__init__(env)
        self.length = length
        self.reinitialize()

    def reinitialize(self):
        self.buffer = np.empty((*self.env.observation_space.shape, self.length), dtype = self.env.observation_space.dtype)
        self.nextup = 0

    def append(self, observation):
        self.buffer[:, :, self.nextup] = observation
        self.nextup += 1
        self.nextup %= self.length

    def reset(self):
        observation = self.env.reset()
        self.append(observation)
        return self.observe()

    def observe(self):
        return self.buffer.copy()

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        return self.observe(), reward, terminal, info

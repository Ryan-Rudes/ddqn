from gym import Wrapper

import numpy as np
import skvideo.io

class VideoRecorder(Wrapper):
    def __init__(self, env, fps=30, show_on_highscore=False):
        super().__init__(env)
        self.frames = []
        self.highscore = 0
        self.fps = fps
        self.viewer = None
        self.show_on_highscore = show_on_highscore

    def show(self):
        if self.viewer is None:
            from gym.envs.classic_control.rendering import SimpleImageViewer
            self.viewer = SimpleImageViewer()

        for frame in self.frames:
            self.viewer.imshow(frame)
        self.viewer.close()

    def save(self):
        if self.show_on_highscore:
            self.show()
        outputdata = np.array(self.frames)
        outputdata = outputdata.astype(np.uint8)
        skvideo.io.vwrite(f'./{self.env.spec.id}-{self.highscore}.mp4', outputdata, inputdict = {'-r': str(self.fps)}, outputdict = {'-pix_fmt': 'yuv420p', '-vcodec': 'libx264', '-r': str(self.fps)})
        self.frames.clear()

    def reset(self):
        self.score = 0
        self.frames.clear()
        observation = self.env.reset()
        self.frames.append(observation)
        return observation

    def step(self, action):
        observation, reward, terminal, info = self.env.step(action)
        self.frames.append(observation)
        self.score += reward
        if self.score > self.highscore:
            self.highscore = self.score
            self.save()
        return observation, reward, terminal, info

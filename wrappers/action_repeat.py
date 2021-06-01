from gym import Wrapper

class ActionRepeat(Wrapper):
    def __init__(self, env, repeat=4):
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0

        for i in range(self.repeat):
            observation, reward, terminal, info = self.env.step(action)
            total_reward += reward

            if terminal:
                break

        return observation, total_reward, terminal, info

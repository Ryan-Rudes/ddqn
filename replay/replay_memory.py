from replay.memory import Memory
import numpy as np

class ReplayMemory:
    def __init__(self, maxlen, shape, dtype=np.uint8):
        self.maxlen = maxlen
        self.nextup = 0
        self.length = 0
        self.rndgen = np.random.default_rng()

        self.shape = shape

        self.states = Memory(maxlen * 2, shape, dtype = dtype)
        self.actions = Memory(maxlen, None, dtype = int)
        self.rewards = Memory(maxlen, None, dtype = float)
        self.terminals = Memory(maxlen, None, dtype = bool)

    def __len__(self):
        return len(self.actions)

    def append(self, state, action, next_state, reward, terminal):
        self.states.extend([state, next_state])
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def sample(self, n, replace = True):
        if replace == 'auto':
            replace = len(self) < n

        indices = self.rndgen.choice(len(self), size = n, replace = replace)

        states = []
        actions = []
        next_states = []
        rewards = []
        terminals = []

        for index in indices:
            states.append(self.states[index * 2])
            actions.append(self.actions[index])
            next_states.append(self.states[index * 2 + 1])
            rewards.append(self.rewards[index])
            terminals.append(self.terminals[index])

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)

        return states, actions, next_states, rewards, terminals

    def exceeds(self, size):
        return len(self) >= size

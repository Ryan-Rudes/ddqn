from extras.transition import Transition
from replay.per.sum_tree import SumTree
from collections.abc import Iterable
from replay.memory import Memory

import numpy as np

class ProportionalPER:
    def __init__(self,
                 maxlen,
                 eps   = 0.01,
                 alpha = 0.6,
                 beta  = 0.4,
                 inc   = 0.0000625):
        self.maxlen = maxlen
        self.memory = SumTree(maxlen)

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.inc = inc

    def __len__(self):
        return len(self.memory)

    def _compute_priority(self, error):
        return (abs(error) + self.eps) ** self.alpha

    def append(self, state, action, next_state, reward, terminal, error):
        transition = Transition(state, action, next_state, reward, terminal)
        p = self._compute_priority(error)
        self.memory.add(error, transition)

    def sample(self, n):
        states = []
        actions = []
        next_states = []
        rewards = []
        terminals = []
        indices = []
        priorities = []

        segment = self.memory.total() / n

        self.beta = min(1, self.beta + self.inc)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)

            idx, p, transition = self.memory.get(s)

            states.append(transition.state)
            actions.append(transition.action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            terminals.append(transition.terminal)
            indices.append(idx)
            priorities.append(p)

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
        priorities = np.array(priorities)

        weights = (priorities / self.memory.total() * len(self.memory)) ** -self.beta
        weights /= weights.max()

        return states, actions, next_states, rewards, terminals, indices, weights

    def update(self, index, error):
        if not isinstance(index, Iterable): index = [index]
        if not isinstance(error, Iterable): error = [error]

        if len(index) != len(error):
            raise ValueError("Number of specified indices and error values do not match")

        for idx, err in zip(index, error):
            p = self._compute_priority(err)
            self.memory.update(idx, p)

    def exceeds(self, size):
        return len(self) >= size

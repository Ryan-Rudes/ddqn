from extras.transition import Transition
from replay.memory import Memory
from itertools import count
import heapq

class PrioritizedBinaryHeap:
    def __init__(self, maxlen, shape):
        self.maxlen = maxlen
        self.memory = Memory(maxlen, None, dtype = tuple)

        self.shape = shape
        self.tiebreaker = count()

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, next_state, reward, terminal, error=0):
        transition = Transition(state, action, next_state, reward, terminal)
        self.memory.append((-error, next(self.tiebreaker), transition))
        heapq._siftdown(self.memory, 0, len(self.memory) - 1)

    def sample(self, n):
        minibatch = [transition for (_, _, transition) in heapq.nsmallest(n, self.memory)]

    def exceeds(self, size):
        return len(self) >= size

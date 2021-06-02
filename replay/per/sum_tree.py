import numpy as np

class SumTree:
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(2 * size - 1)
        self.data = np.empty(size, dtype = object)
        self.indx = 0
        self.length = 0

    def __len__(self):
        return self.length

    def _propagate(self, idx, delta):
        while True:
            idx = (idx - 1) >> 1
            self.tree[idx] += delta

            if idx == 0:
                break

    def _retrieve(self, idx, s):
        l = 2 * idx + 1
        r = l + 1

        if l >= len(self.tree):
            return idx

        if s <= self.tree[l]:
            return self._retrieve(l, s)
        else:
            return self._retrieve(r, s - self.tree[l])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.indx + self.size - 1
        self.data[self.indx] = data
        self.update(idx, p)

        self.indx += 1
        self.indx %= self.size

        self.length = min(self.length + 1, self.size)

    def update(self, idx, p):
        delta = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, delta)

    def get(self, s):
        idx = self._retrieve(0, s)
        return (idx, self.tree[idx], self.data[idx - self.size + 1])

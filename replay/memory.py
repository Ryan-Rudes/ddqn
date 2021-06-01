import numpy as np

class Memory:
    def __init__(self, maxlen, shape, dtype=np.uint8):
        if shape is None:
            shape = ()

        self.maxlen = maxlen
        self.nextup = 0
        self.length = 0
        self.buffer = np.empty((maxlen, *shape), dtype = dtype)
        self.rndgen = np.random.default_rng()

        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, i):
        return self.buffer[i]

    def __len__(self):
        return self.length

    def append(self, image):
        self.buffer[self.nextup] = image
        self.nextup += 1
        self.nextup %= self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def extend(self, images):
        for image in images:
            self.append(image)

    def sample(self, n, replace=False):
        if replace == 'auto':
            replace = self.length < n

        indices = self.rndgen.choice(self.length, size = n, replace = replace)
        return [self.buffer[idx] for idx in indices]

    def exceeds(self, size):
        return self.length >= size

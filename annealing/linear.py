from annealing.schedule import Schedule

class LinearDecaySchedule(Schedule):
    def __init__(self, mineps=0.01, maxeps=1.0, length=100000):
        self.mineps = mineps
        self.maxeps = maxeps
        self.length = length

        self.epsilon = maxeps
        self.decay = (maxeps - mineps) / length

    def update(self):
        self.epsilon = max(self.mineps, self.epsilon - self.decay)

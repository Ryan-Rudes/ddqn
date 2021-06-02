from annealing.schedule import Schedule

class ExponentialDecaySchedule(Schedule):
    def __init__(self, mineps=0.01, maxeps=1.0, decay=0.999):
        self.mineps = mineps
        self.maxeps = maxeps

        self.epsilon = maxeps
        self.decay = decay

    def update(self):
        self.epsilon = max(self.mineps, self.epsilon * self.decay)

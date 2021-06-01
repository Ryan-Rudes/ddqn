from tensorflow.keras.models import *
from tensorflow.keras.layers import *

class NatureDQN(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.num_actions = num_actions

        self.conv1 = Conv2D(32, 8, 4, activation = 'relu')
        self.conv2 = Conv2D(64, 4, 2, activation = 'relu')
        self.conv3 = Conv2D(64, 3, 1, activation = 'relu')

        self.flatten = Flatten()

        self.dense1 = Dense(512, activation = 'relu')
        self.dense2 = Dense(num_actions)

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        q = self.dense2(x)
        return q

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

class CartPoleDQN(Sequential):
    def __init__(self):
        super().__init__([
            Input((4,)),
            Dense(24, activation = 'relu'),
            Dense(24, activation = 'relu'),
            Dense(2)
        ])

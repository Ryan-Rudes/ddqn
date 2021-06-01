from torch import nn

class NatureDQN(nn.Module):
    def __init__(self, num_actions):
        Super().__init__()

        self.num_actions = num_actions
        self.seq = nn.Sequential(
            nn.Conv2d( 1, 32, 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, input):
        return self.seq(input)

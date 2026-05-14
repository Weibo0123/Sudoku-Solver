# model.py
import torch.nn as nn

class ImitationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImitationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

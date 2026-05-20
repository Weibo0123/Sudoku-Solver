# q_network.py
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, board_size, output_size):
        super(QNetwork, self).__init__()
        self.board_size = board_size

        self.conv = nn.Sequential(
            nn.Conv2d(board_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (board_size ** 2), 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
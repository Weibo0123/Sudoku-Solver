# model.py
import torch.nn as nn


class ImitationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(81, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.cell_head = nn.Linear(256, 81)
        self.num_head  = nn.Linear(256, 9)

    def forward(self, x):
        feat = self.backbone(x)
        return self.cell_head(feat), self.num_head(feat)
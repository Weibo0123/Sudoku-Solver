# model.py
import torch
import torch.nn as nn


class ImitationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
        )
        self.cell_head = nn.Linear(128 * 81, 81)
        self.num_head  = nn.Linear(128, 9)

    def forward(self, x, cell_label=None):
        x_2d = x.view(-1, 1, 9, 9)
        feat = self.cnn(x_2d)
        feat = feat.view(x.size(0), 128, 81)
        cell_logits = self.cell_head(feat.view(x.size(0), -1))

        if cell_label is not None:
            idx = cell_label
        else:
            idx = cell_logits.argmax(dim=1)

        cell_feat = feat[torch.arange(feat.size(0)), :, idx]
        num_logits = self.num_head(cell_feat)

        return cell_logits, num_logits
# model.py
import torch
import torch.nn as nn


class ImitationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 保持 9x9
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
        )
        self.cell_head = nn.Linear(128 * 81, 81)
        self.num_head  = nn.Linear(128 * 81 + 81, 9)

    def forward(self, x, cell_label=None):
        x = x.view(-1, 1, 9, 9)
        feat = self.cnn(x).view(x.size(0), -1)
        cell_logits = self.cell_head(feat)

        if cell_label is not None:
            cell_onehot = torch.zeros(x.size(0), 81).to(x.device)
            cell_onehot.scatter_(1, cell_label.unsqueeze(1), 1.0)
        else:
            cell_onehot = torch.softmax(cell_logits, dim=1)

        num_logits = self.num_head(torch.cat([feat, cell_onehot], dim=1))
        return cell_logits, num_logits

# critic_network.py
import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(
            self,
            embed_dim: int = 32,
            num_heads: int = 4,
            num_layers: int = 2,
            ffn_dim: int = 64,
            dropout: float = 0.1
    ):
        super().__init__()

        self.value_embedding = nn.Embedding(10, embed_dim)

        self.row_embedding = nn.Embedding(9, embed_dim)
        self.col_embedding = nn.Embedding(9, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, 1))

        rows = torch.arange(81) // 9
        cols = torch.arange(81) % 9
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)


    def forward(self, board: torch.Tensor) -> torch.Tensor:
        B = board.size(0)

        x = (
            self.value_embedding(board)
            + self.row_embedding(self.rows.unsqueeze(0).expand(B, -1))
            + self.col_embedding(self.cols.unsqueeze(0).expand(B, -1))
        )

        x = self.transformer(x)

        x = x.mean(dim=1)

        value = self.output_head(x)
        return value
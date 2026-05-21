# agent_network.py
import torch
import torch.nn as nn
import math

AGENT_ROW = 0
AGENT_COL = 1
AGENT_BOX = 2

class AgentTransformer(nn.Module):
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, num_layers: int = 2, ffn_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.value_embedding = nn.Embedding(10, embed_dim)
        self.position_embedding = nn.Embedding(9, embed_dim)
        self.agent_type_embedding = nn.Embedding(3, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, norm_first=True)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_head = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, 9))


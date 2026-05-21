import torch
from marl.agent_network import AgentTransformer, AGENT_ROW, AGENT_COL, AGENT_BOX

class AgentManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.agent_network = AgentTransformer().to(device)

        self.correct_counts = [0] * 27
        self.total_counts   = [0] * 27



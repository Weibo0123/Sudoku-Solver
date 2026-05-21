import torch
from marl.agent_network import AgentTransformer, AGENT_ROW, AGENT_COL, AGENT_BOX

class AgentManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.agent_network = AgentTransformer().to(device)

        self.correct_counts = [0] * 27
        self.total_counts   = [0] * 27

    def row_agent_idx(self, row: int) -> int:
        return row

    def col_agent_idx(self, col: int) -> int:
        return 9 + col

    def box_agent_idx(self, box: int) -> int:
        return 18 + box

    def box_idx(self, row: int, col: int) -> int:
        return (row // 3) * 3 + (col // 3)

    def update_accuracy(self, agent_idx: int, correct: bool):
        self.total_counts[agent_idx] += 1
        if correct:
            self.correct_counts[agent_idx] += 1

    def get_accuracy(self, agent_idx: int) -> float:
        if self.total_counts[agent_idx] == 0:
            return 0.5
        return self.correct_counts[agent_idx] / self.total_counts[agent_idx]

    def reset_accuracy(self):
        self.correct_counts = [0] * 27
        self.total_counts   = [0] * 27



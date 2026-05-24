# arbitrator.py
import torch

class Arbitrator:
    def __init__(self, device="cpu"):
        self.device = device
        self.remaining = [9] * 27

    def reset(self, board: list):
        size = len(board)
        box_size = int(size ** 0.5)

        for row in range(size):
            self.remaining[row] = sum(1 for c in range (size) if board[row][c] == 0)

        for col in range(size):
            self.remaining[9 + col] = sum(1 for r in range(size) if board[r][col] == 0)

        for box in range(size):
            start_row = box // box_size * box_size
            start_col = box % box_size * box_size
            count = sum(
                1
                for r in range(start_row, start_row + box_size)
                for c in range(start_col, start_col + box_size)
                if board[r][c] == 0
            )
            self.remaining[18 + box] = count

    def update_remaining(self, agent_idx: dict):
        for idx in agent_idx.values():
            self.remaining[idx] = max(0, self.remaining[idx] - 1)

    def _compute_weight(self, agent_idc: int, accuracy: float):
        remaining = self.remaining[agent_idc]
        if remaining == 0:
            raise ValueError(f"Agent {agent_idc} has remaining=0 but is still being queried. This should never happen.")
        if remaining == 1:
            return float("inf")
        return accuracy * (1.0 / remaining)

    def decide(
            self,
            scores: dict,
            agent_idx: dict,
            agent_manger,
            action_mask: torch.Tensor,
    ) -> int:
        weighted_scores = torch.zeros(9).to(self.device)
        for agent_type in ("row", "col", "box"):
            idx = agent_idx[agent_type]
            accuracy = agent_manger.get_accuracy(idx)
            weight = self._compute_weight(idx, accuracy)
            weighted_scores += weight * scores[agent_type].to(self.device)

        weighted_scores = torch.nan_to_num(weighted_scores, nan=0.0, posinf=1e9, neginf=-1e9)

        weighted_scores[~action_mask.to(self.device)] = float("-inf")

        if not action_mask.any():
            return None

        digit = weighted_scores.argmax().item() + 1
        return digit
# arbitrator.py
import torch

class Arbitrator:
    def __init__(self):
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
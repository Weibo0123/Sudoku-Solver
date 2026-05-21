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

    def _extract_row_input(self, board: list, row: int):
        cell_value = [board[row][c] for c in range(9)]
        position = list(range(9))
        return cell_value, position

    def _extract_col_input(self, board: list, col: int):
        cell_value = [board[r][col] for r in range(9)]
        position = list(range(9))
        return cell_value, position

    def _extract_box_input(self, board: list, box: int):
        start_row = box // 3 * 3
        start_col = box % 3 * 3
        cell_value = []
        position = []
        pos = 0
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                cell_value.append(board[r][c])
                position.append(pos)
                pos += 1
        return cell_value, position

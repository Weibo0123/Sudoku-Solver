# agent_manager.py
import torch

from marl.agent_network import AgentTransformer, AGENT_ROW, AGENT_COL, AGENT_BOX
from env.sudoku_env import SudokuEnv
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

    def get_actor_inputs(self, board, target_row, target_col):
        box = self.box_agent_idx(target_row)
        row_cells, row_pos = self._extract_row_input(board, target_row)
        col_cells, col_pos = self._extract_col_input(board, target_col)
        box_cells, box_pos = self._extract_box_input(board, box)
        return row_cells, row_pos, col_cells, col_pos, box_cells, box_pos

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

    def _get_action_mask(self, env, row: int, col: int) -> list:
        return [env.is_valid(row=row, col=col, value=digit) for digit in range(1, 10)]

    def get_logits_with_grad(self, env, board, target_row, target_col):
        box = self.box_idx(target_row, target_col)
        row_cells, row_pos = self._extract_row_input(board, target_row)
        col_cells, col_pos = self._extract_col_input(board, target_col)
        box_cells, box_pos = self._extract_box_input(board, box)
        action_mask = self._get_action_mask(env, target_row, target_col)

        cell_values_t = torch.tensor(
            [row_cells, col_cells, box_cells], dtype=torch.long
        ).to(self.device)
        positions_t = torch.tensor(
            [row_pos, col_pos, box_pos], dtype=torch.long
        ).to(self.device)
        agent_types_t = torch.tensor(
            [AGENT_ROW, AGENT_COL, AGENT_BOX], dtype=torch.long
        ).to(self.device)
        mask_t = torch.stack([
                                 torch.tensor(action_mask, dtype=torch.bool)
                             ] * 3).to(self.device)

        logits = self.agent_network(cell_values_t, positions_t, agent_types_t, mask_t)
        return logits, cell_values_t, positions_t, agent_types_t, mask_t

    def get_scores(self, env, board: list, target_row: int, target_col: int):
        action_mask = self._get_action_mask(env, target_row, target_col)

        with torch.no_grad():
            logits, _, _, _, _ = self.get_logits_with_grad(env, board, target_row, target_col)

        scores = {
            "row": logits[0],
            "col": logits[1],
            "box": logits[2],
        }

        agent_indices = {
            "row": self.row_agent_idx(target_row),
            "col": self.col_agent_idx(target_col),
            "box": self.box_agent_idx(self.box_idx(target_row, target_col)),
        }

        return scores, agent_indices, torch.tensor(action_mask, dtype=torch.bool)

    def parameters(self):
        return self.agent_network.parameters()

# sudoku_env
import numpy as np

class SudokuEnv:
    def __init__(self, board):
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)

        self.initial_board = [row[:] for row in board]
        self.board = [row[:] for row in board]
        self.steps = 0
        empty_count = sum(1 for row in board for cell in row if cell == 0)
        self.max_steps = empty_count * 2

    def reset(self):
        self.board = [row[:] for row in self.initial_board]
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return [row[:] for row in self.board]

    def is_valid(self, row, col, value):
        # Check the row
        if any(self.board[row][c] == value for c in range(9)):
            return False

        # Check the column
        if any(self.board[r][col] == value for r in range(9)):
            return False

        # Check 3×3 box
        start_row = row // 3 * 3
        start_col = col // 3 * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == value:
                    return False

        return True

    def apply_action(self, action):
        row, col, value = action
        # If the cell is empty and the input is valid, apply the number to the board
        if self.board[row][col] == 0 and self.is_valid(row, col, value):
            self.board[row][col] = value
            return True
        return False

    def get_empty_cell(self):
        empty = []
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        return empty

    def is_solved(self):
        target = set(range(1, 10))
        for i in range(9):
            if set(self.board[i]) != target:
                return False
            if set(self.board[r][i] for r in range(9)) != target:
                return False
        for br in range(3):
            for bc in range(3):
                box = {self.board[br * 3 + r][bc * 3 + c] for r in range(3) for c in range(3)}
                if box != target:
                    return False
        return True

    def get_valid_actions(self):
        actions = []

        empty_cells = self.get_empty_cell()
        for r, c in empty_cells:
            for v in range(1, 10):
                if self.is_valid(r, c, v):
                    actions.append((r, c, v))

        return actions

    def step(self, action):
        row, col, value = action
        self.steps += 1

        reward = 0
        done = False

        if self.board[row][col] != 0:
            reward = -1
            return self.get_state(), reward, done, {}

        if not self.is_valid(row, col, value):
            reward = -1
            return self.get_state(), reward, done, {}

        self.board[row][col] = value
        reward = 0.1

        if self.is_solved():
            reward = 10
            done = True
        elif self.steps >= self.max_steps:
            done = True
            reward = -5

        return self.get_state(), reward, done, {}

    def print_board(self):
        for r in range(9):
            if r % 3 == 0:
                print("+-------+-------+-------+")
            row_str = "| "
            for c in range(9):
                val = self.board[r][c] if self.board[r][c] != 0 else "."
                row_str += str(val) + " "
                if (c + 1) % 3 == 0:
                    row_str += "| "
            print(row_str)
        print("+-------+-------+-------+")



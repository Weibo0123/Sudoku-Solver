# sudoku_env

class SudokuEnv:
    def __init__(self, board):
        self.size = len(board)
        self.box_size = int(self.size ** 0.5)

        assert self.box_size ** 2 == self.size

        self.initial_board = [row[:] for row in board]
        self.board = [row[:] for row in board]
        self.steps = 0
        empty_count = sum(1 for row in board for cell in row if cell == 0)
        self.max_steps = empty_count * 2

        self.history = []

    def reset(self):
        self.board = [row[:] for row in self.initial_board]
        self.steps = 0
        return self.get_state()

    def get_state(self):
        return [row[:] for row in self.board]

    def is_valid(self, row, col, value):
        # Check the row
        if any(self.board[row][c] == value for c in range(self.size)):
            return False

        # Check the column
        if any(self.board[r][col] == value for r in range(self.size)):
            return False

        # Check 3×3 box
        start_row = row // self.box_size * self.box_size
        start_col = col // self.box_size * self.box_size
        for i in range(start_row, start_row + self.box_size):
            for j in range(start_col, start_col + self.box_size):
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
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        return empty

    def is_solved(self):
        target = set(range(1, self.size + 1))
        for i in range(self.size):
            if set(self.board[i]) != target:
                return False
            if set(self.board[r][i] for r in range(self.size)) != target:
                return False
        for br in range(self.box_size):
            for bc in range(self.box_size):
                box = {self.board[br * self.box_size + r][bc * self.box_size + c] for r in range(self.box_size) for c in range(self.box_size)}
                if box != target:
                    return False
        return True

    def get_valid_actions(self):
        actions = []

        empty_cells = self.get_empty_cell()
        for r, c in empty_cells:
            for v in range(1, self.size + 1):
                if self.is_valid(r, c, v):
                    actions.append((r, c, v))
        return actions

    def push(self, action):
        row, col, value = action
        self.history.append((row, col, self.board[row][col]))  # 记录原来的值

    def undo(self):
        if not self.history:
            return False
        row, col, original = self.history.pop()
        self.board[row][col] = original
        self.steps -= 1
        return True

    def step(self, action):
        row, col, value = action
        self.steps += 1
        reward = 0
        done = False

        if self.board[row][col] != 0:
            reward = -0.1
            return self.get_state(), reward, done, {}

        if not self.is_valid(row, col, value):
            reward = -0.1
            return self.get_state(), reward, done, {}

        self.board[row][col] = value
        reward = 0.01

        if self.is_solved():
            reward = 1
            done = True
        elif self.steps >= self.max_steps:
            done = True
            reward = -0.5

        return self.get_state(), reward, done, {}

    def print_board(self):
        for r in range(self.size):
            if r % self.box_size == 0:
                print("+".join(["-" * (self.box_size * 2 + 1)] * self.box_size))
            row_str = " "
            for c in range(self.size):
                if c % self.box_size == 0:
                    row_str += "| "
                val = self.board[r][c] if self.board[r][c] != 0 else "."
                row_str += str(val) + " "
            row_str += "|"
            print(row_str)
        print("+".join(["-" * (self.box_size * 2 + 1)] * self.box_size))




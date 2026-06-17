# sudoku_env
"""
A module for managing a Sudoku environment, supporting board validation, actions, and game state updates.

This module provides the `SudokuEnv` class, which defines the behavior of a Sudoku game environment. It includes
methods to reset the game, check for valid moves, apply actions, undo moves, and determine if the board has been
solved. The class also allows step-by-step interaction with the board, including tracking the history of applied actions
and providing valid actions based on the current state of the board.
"""
class SudokuEnv:
    def __init__(self, board, solution=None):
        self.size = len(board)
        self.solution = solution
        self.box_size = int(self.size ** 0.5)

        assert self.box_size ** 2 == self.size

        self.initial_board = [row[:] for row in board]
        self.board = [row[:] for row in board]
        self.steps = 0
        empty_count = sum(1 for row in board for cell in row if cell == 0)
        self.max_steps = empty_count * 20

        self.history = []

    def reset(self):
        """
        Resets the current game state to its initial configuration.

        Returns:
            list: The current state of the game board after reset.
        """
        self.board = [row[:] for row in self.initial_board]
        self.steps = 0
        self.history = []
        return self.get_state()

    def get_state(self):
        """
        Creates and returns a copy of the current state of the board. The state is
        represented as a list of lists.

        Returns:
            list: A deep copy of the current board state.
        """
        return [row[:] for row in self.board]

    def is_valid(self, row, col, value):
        """
        Determines the validity of placing a value in a specified cell on the board according to the constraints
        of rows, columns, and subgrid (3x3) uniqueness in the context of a Sudoku board.

        Parameters:
        row: int
            The row index of the cell where the value is being placed.
        col: int
            The column index of the cell where the value is being placed.
        value: int
            The value to be placed in the specified cell.

        Returns:
        bool
            True if placing the value in the specified cell is valid; otherwise, False.
        """
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
        """
        Applies a given action to the game board if the action is valid.

        Parameters:
            action (tuple): A tuple consisting of the row (int), column (int),
                and value (int) to be applied to the board.

        Returns:
            bool: True if the action was successfully applied to the board,
                otherwise False.
        """
        row, col, value = action
        # If the cell is empty and the input is valid, apply the number to the board
        if self.board[row][col] == 0 and self.is_valid(row, col, value):
            self.board[row][col] = value
            return True
        return False

    def get_empty_cell(self):
        """
        Find and extract all empty cells from the game board.

        Returns:
            list: A list of tuples where each tuple contains the row and column indexes of an empty cell.
        """
        empty = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty.append((r, c))
        return empty

    def is_solved(self):
        """
        Determine if the current Sudoku board is solved.

        A solved Sudoku board must satisfy the following conditions:
        1. Every row contains all numbers from 1 to `size` with no duplicates.
        2. Every column contains all numbers from 1 to `size` with no duplicates.
        3. Every sub-grid (box) contains all numbers from 1 to `size` with no duplicates.

        Returns:
            bool: True if the board is solved, False otherwise.
        """
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
        """
        Generates a list of valid actions based on the current state of the board.

        Returns:
            list: A list of tuples where each tuple represents a valid action in the
            format (row, column, value).
        """
        actions = []

        empty_cells = self.get_empty_cell()
        for r, c in empty_cells:
            for v in range(1, self.size + 1):
                if self.is_valid(r, c, v):
                    actions.append((r, c, v))
        return actions

    def push(self, action):
        """
        Pushes an action to the history stack and updates the internal state.

        Parameters:
            action (tuple): A tuple containing the row index, column index, and the new value
            to be set in the board. The row and column indices represent the position of the
            cell on the board.

        Args:
            action: A tuple containing (row, col, value). The `row` and `col` are integers
            specifying the position on the board. `value` represents the new value intended
            for the position.
        """
        row, col, value = action
        self.history.append((row, col, self.board[row][col]))

    def undo(self):
        """
        Reverts the last change made to the board by undoing the most recent action in the history.

        Returns:
            bool: True if an action was undone, False if there was no history to undo.
        """
        if not self.history:
            return False
        row, col, original = self.history.pop()
        self.board[row][col] = original
        self.steps -= 1
        return True

    def step(self, action):
        """
        Performs a single step in the game simulation by applying the given action.

        Parameters:
        action : tuple
            A tuple containing row, column, and the value to be placed on the board.

        Returns:
        tuple
            A tuple containing the updated board state, a reward of 0, a boolean
            indicating if the game is finished, and an empty dictionary.
        """
        row, col, value = action
        self.steps += 1
        done = False

        if self.board[row][col] != 0:
            return self.get_state(), 0, done, {}

        if not self.is_valid(row, col, value):
            return self.get_state(), 0, done, {}

        self.history.append((row, col, 0))
        self.board[row][col] = value

        if self.is_solved():
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self.get_state(), 0, done, {}

    def print_board(self):
        """
        Prints the current state of the Sudoku board to the console in a human-readable format.
        """
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




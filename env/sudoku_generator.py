#sudoku_generator
import random
from env.sudoku_env import SudokuEnv

def generate_sudoku(board=None, size=9):
    """
    Generates a valid Sudoku board by recursively filling empty cells with valid numbers.

    Args:
        board (list[list[int]]): Optional. A 2D list representing a partially completed Sudoku
                                  board. Defaults to None, in which case an empty board is created.
        size (int): Optional. The size of the Sudoku grid, typically 9 for a standard 9x9 Sudoku.

    Returns:
        list[list[int]] or None: A 2D list representing a completed Sudoku board if successful.
                                 Returns None if no valid board can be generated.
    """
    if board is None:
        board = [[0]*size for _ in range(size)]

    empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]
    if not empty:
        return board

    row, col = empty[0]
    nums = list(range(1, size + 1))
    random.shuffle(nums)

    for num in nums:
        if SudokuEnv(board).is_valid(row, col, num):
            board[row][col] = num
            if generate_sudoku(board, size):
                return board
            board[row][col] = 0
    return None

def create_sudoku(full_board, empty_cells=40, size=9):
    """
    Generates a Sudoku puzzle and its solution by modifying a full (solved) Sudoku board.
    The function takes a complete Sudoku board, removes a specified number of cells to create
    a puzzle, and returns both the puzzle and the solution. The initial board remains unchanged.

    Parameters:
        full_board (list[list[int]]): A 2D list representing a completed Sudoku board where each
            sub-list represents a row of integers from 1 to 9.
        empty_cells (int): The number of cells to remove from the full board to create the
            puzzle (default is 40).
        size (int): The dimensions of the Sudoku board, typically 9 for a 9x9 grid (default is 9).

    Returns:
        tuple: A tuple containing two elements:
            - puzzle (list[list[int]]): A 2D list representing the Sudoku puzzle with several
              empty cells replaced with 0.
            - solution (list[list[int]]): A 2D list representing the original fully solved
              Sudoku board.
    """
    puzzle = [row[:] for row in full_board]
    solution = [row[:] for row in full_board]
    cell = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cell)

    for i in range(empty_cells):
        r, c = cell[i]
        puzzle[r][c] = 0

    return puzzle, solution


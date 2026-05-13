#sudoku_generator
import random
from sudoku_env import SudokuEnv

def generate_sudoku(board=None):
    if board is None:
        board = [[0]*9 for _ in range(9)]

    empty = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    if not empty:
        return board

    row, col = empty[0]
    nums = list(range(1, 10))
    random.shuffle(nums)

    for num in nums:
        if SudokuEnv(board).is_valid(row, col, num):
            board[row][col] = num
            if generate_sudoku(board):
                return board
            board[row][col] = 0
    return None

def create_sudoku(full_board, empty_cells=40):
    puzzle = [row[:] for row in full_board]
    cell = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cell)

    for i in range(empty_cells):
        r, c = cell[i]
        puzzle[r][c] = 0

    return puzzle


#sudoku_generator
import random
from env.sudoku_env import SudokuEnv

def generate_sudoku(board=None, size=9):
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
    puzzle = [row[:] for row in full_board]
    cell = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cell)

    for i in range(empty_cells):
        r, c = cell[i]
        puzzle[r][c] = 0

    return puzzle


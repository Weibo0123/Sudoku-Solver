import pytest
from sudoku_env import SudokuEnv
from sudoku_generator import generate_sudoku, create_sudoku

def test_generate_sudoku_produces_valid_solution():
    board = generate_sudoku()
    assert board is not None

    # All cells must be non-zero
    assert all(board[r][c] != 0 for r in range(9) for c in range(9))

    # All rows must contain 1–9
    for r in range(9):
        assert sorted(board[r]) == list(range(1, 10))

    # All columns must contain 1–9
    for c in range(9):
        col = [board[r][c] for r in range(9)]
        assert sorted(col) == list(range(1, 10))

    # All 3x3 squares must contain 1–9
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            square = [
                board[r][c]
                for r in range(box_row, box_row + 3)
                for c in range(box_col, box_col + 3)
            ]
            assert sorted(square) == list(range(1, 10))


def test_create_sudoku_hides_correct_cells():
    solved = generate_sudoku()
    puzzle = create_sudoku(solved, empty_cells=40)

    # puzzle still valid size
    assert len(puzzle) == 9
    assert all(len(row) == 9 for row in puzzle)

    # count zeros
    zero_count = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] == 0)
    assert zero_count == 40

def test_create_sudoku_does_not_modify_original():
    solved = generate_sudoku()
    solved_copy = [row[:] for row in solved]

    puzzle = create_sudoku(solved, empty_cells=20)

    assert solved == solved_copy
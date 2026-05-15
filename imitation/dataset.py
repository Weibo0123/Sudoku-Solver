# dataset.py
from env.sudoku_generator import generate_sudoku, create_sudoku
from imitation.solve import solve_and_record

def generate_dataset(num_puzzles=1000, size=9, empty_cells=40):
    X, y_cell, y_num = [], [], []

    for _ in range(num_puzzles):
        board = generate_sudoku(size=size)
        puzzle = create_sudoku(board, empty_cells=empty_cells, size=size)
        steps = solve_and_record(puzzle)
        if steps is None:
            continue

        for board_state, (row, col, num) in steps:
            X.append([cell / 9.0 for r in board_state for cell in r])
            y_cell.append(row * 9 + col)
            y_num.append(num - 1)

    return X, y_cell, y_num
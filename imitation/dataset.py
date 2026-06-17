# dataset.py
from env.sudoku_generator import generate_sudoku, create_sudoku
from imitation.solve import solve_and_record

def generate_dataset(num_puzzles=1000, size=9, empty_cells=40):
    """
    Generates a dataset of sudoku puzzles and their corresponding solving steps.

    Parameters:
    num_puzzles: int, optional
        The number of sudoku puzzles to generate and record. Defaults to 1000.
    size: int, optional
        The dimensional size of the sudoku board (e.g., 9 for standard 9x9 sudoku).
        Defaults to 9.
    empty_cells: int, optional
        The number of empty cells in each generated puzzle. Defaults to 40.

    Returns:
    tuple
        A tuple containing three elements:
        - X: List of float
            A list of normalized board states, where the board cells are divided by 9.0.
        - y_cell: List of int
            A list of cell indices for the target cell to fill during solving.
        - y_num: List of int
            A list of numbers to fill corresponding to the target cell indices.
    """
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
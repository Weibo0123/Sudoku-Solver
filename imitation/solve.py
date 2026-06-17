# solve.py

def is_valid(board, row, col, value):
    """
    Checks the validity of placing a value in a specific cell of the Sudoku board.

    Parameters:
    board : list[list[int]]
        The Sudoku board represented as a 2D list of integers.
    row : int
        The row index where the value is to be placed.
    col : int
        The column index where the value is to be placed.
    value : int
        The value to be checked for placement.

    Returns:
    bool
        True if the value can be legally placed in the specified cell of the
        Sudoku board, otherwise False.
    """
    size = len(board)
    box_size = int(size ** 0.5)

    if any(board[row][c] == value for c in range (size)):
        return False
    if any(board[r][col] == value for r in range (size)):
        return False

    start_row = row // box_size * box_size
    start_col = col // box_size * box_size
    for i in range(start_row, start_row + box_size):
        for j in range(start_col, start_col + box_size):
            if board[i][j] == value:
                return False
    return True


def solve_and_record(board):
    """
    Solve the Sudoku puzzle and record each step of the solving process.

    Parameters:
        board: list[list[int]]
            A 9x9 grid representing the Sudoku puzzle where empty cells are denoted by 0.

    Returns:
        list[tuple[list[list[int]], tuple[int, int, int]]] | None
            A list of tuples capturing the solving steps. Each tuple contains:
            - A snapshot of the board as a 9x9 grid.
            - A tuple (row, col, num) identifying the cell and the number placed.
            If the puzzle is unsolvable, returns None.
    """
    if board is None:
        return None

    steps = []

    def backtrack():
        empty = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
        if not empty:
            return True

        row, col = empty[0]
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                snapshot = [r[:] for r in board]
                board[row][col] = num
                steps.append((snapshot, (row, col, num)))
                if backtrack():
                    return True
                board[row][col] = 0
        return False

    if not backtrack():
        return None
    return steps
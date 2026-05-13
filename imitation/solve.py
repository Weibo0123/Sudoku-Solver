# solve.py

def is_valid(board, row, col, value):
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


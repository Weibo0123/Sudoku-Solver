from env.sudoku_env import SudokuEnv

# Sample boards
empty_board = [[0]*9 for _ in range(9)]
partially_filled_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

def test_is_valid():
    env = SudokuEnv(empty_board)
    # Any number should be valid in an empty board
    for r in range(9):
        for c in range(9):
            for v in range(1, 10):
                assert env.is_valid(r, c, v)

    env = SudokuEnv(partially_filled_board)
    # Placing 5 at (0,2) should be invalid because 5 is in the same row
    assert env.is_valid(0, 2, 5) == False
    # Placing 2 at (0,2) should be valid
    assert env.is_valid(0, 2, 2)
    # Placing 6 at (2,0) should be invalid because 6 is in the same column
    assert env.is_valid(2, 0, 6) ==  False

def test_get_empty_cell():
    env = SudokuEnv(empty_board)
    empty_cells = env.get_empty_cell()
    # All 81 cells should be empty
    assert len(empty_cells) == 81

    env = SudokuEnv(partially_filled_board)
    empty_cells = env.get_empty_cell()
    # Some cells are filled
    assert len(empty_cells) == 51
    # All returned cells should indeed be empty
    for r, c in empty_cells:
        assert env.board[r][c] == 0

def test_apply_action():
    env = SudokuEnv(empty_board)
    # Apply a valid action
    assert env.apply_action((0,0,1)) == True
    assert env.board[0][0] == 1
    # Apply an invalid action (cell already filled)
    assert env.apply_action((0,0,2)) == False
    # Apply an invalid action (violates Sudoku rules)
    env.board[0][1] = 1
    assert env.apply_action((0,2,1)) == False

def test_is_solved():
    env = SudokuEnv(empty_board)
    assert env.is_solved() == False

    solved_board = [
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9]
    ]
    env = SudokuEnv(solved_board)
    assert env.is_solved() == True

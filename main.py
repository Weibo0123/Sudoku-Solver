# main.py

from sudoku_env import SudokuEnv
from sudoku_generator import generate_sudoku, create_sudoku

def main():
    # Get the difficulty from user input
    empty_cells = print_menu()

    # Generate a full Sudoku board
    full_board = generate_sudoku()

    # Create a puzzle with the specified number of empty cells
    puzzle = create_sudoku(full_board, empty_cells=empty_cells)

    # Initialize the environment and print the puzzle
    env = SudokuEnv(puzzle)
    print(f"\nSudoku Puzzle ({empty_cells} empty cells):")

    play_sudoku(env)


def print_menu():
    # Prompt the player to choose difficulty
    print("Select Sudoku difficulty:")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")
    print("4. Expert")

    # Get player input
    choice = input("Enter difficulty number (1-4): ").strip()

    # Map choice to number of empty cells
    difficulty_map = {
        "1": 40,  # Easy
        "2": 50,  # Medium
        "3": 60,  # Hard
        "4": 70   # Expert
    }

    empty_cells = difficulty_map.get(choice, 40)  # Default to Easy
    return empty_cells

def play_sudoku(env):
    """
    Interactive loop to let the player fill the Sudoku board.
    Player inputs: row, column, number.
    Type 'q' to quit.
    """
    while True:
        print("\nCurrent board:")
        env.print_board()

        user_input = input("Enter your move as row,col,num (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            print("Exiting game.")
            break

        try:
            row, col, num = map(int, user_input.split(','))
            # Convert 1-based input to 0-based index
            row -= 1
            col -= 1

            if env.board[row][col] != 0:
                print("Cell is already filled. Try another cell.")
                continue

            if env.is_valid(row, col, num):
                env.board[row][col] = num
                print(f"Placed {num} at ({row+1}, {col+1})")
            else:
                print(f"{num} cannot be placed at ({row+1}, {col+1}). Invalid move.")

            # Check if the puzzle is completed
            if all(all(cell != 0 for cell in row_cells) for row_cells in env.board):
                print("\nCongratulations! You have completed the Sudoku puzzle!")
                env.print_board()
                break

        except ValueError:
            print("Invalid input format. Use row,col,num (e.g., 0,1,5)")
        except IndexError:
            print("Row and column must be between 0 and 8.")

if __name__ == '__main__':
    main()

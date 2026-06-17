import time
import os
import torch
import sys
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from imitation.model import ImitationModel
from imitation.train import solve_with_mrv, train


def resource_path(relative_path):

    try:
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_board(state, initial_board=None, highlight=None):
    """
    Print the board.
    """
    print("+" + "---------+" * 3)
    for i, row in enumerate(state):
        line = "|"
        for j, val in enumerate(row):
            is_initial = initial_board and initial_board[i][j] != 0
            is_highlight = highlight == (i, j)

            if val == 0:
                cell = "*"
            else:
                cell = str(val)

            if is_highlight:
                cell = f"[{cell}]"   # bracket marks the conflicting cell
            elif is_initial:
                cell = f" {cell} "
            else:
                cell = f"{cell}"   # asterisks mark player-entered digits

            line += cell.center(3)
            if (j + 1) % 3 == 0:
                line += "|"
        print(line)
        if (i + 1) % 3 == 0:
            print("+" + "---------+" * 3)


def load_model(device):
    """
    Loads a pre-trained Sudoku model, trains a new model if no checkpoint
    exists, and prepares it for evaluation.

    Args:
        device: The device on which the model will be loaded (e.g., CPU or GPU).

    Returns:
        The loaded Sudoku model prepared for evaluation.
    """
    model_path = resource_path('checkpoints/sudoku_model.pth')
    if not os.path.exists(model_path):
        print("No checkpoint found. Starting training...")
        train()
        print("Training complete!\n")
    model = ImitationModel()
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
#  AI Mode
# ──────────────────────────────────────────────
def run_ai_mode(model):
    """
    Executes the AI Solver Mode to solve a Sudoku puzzle.

    Parameters:
        model (callable): The AI model used to solve the puzzle. It should be a callable
        object that follows the required interface for puzzle-solving.
    """
    clear()
    print("=== AI Solver Mode ===\n")

    full_board = generate_sudoku(size=9)
    puzzle, solution = create_sudoku(full_board, empty_cells=40, size=9)
    env = SudokuEnv(puzzle, solution)
    env.reset()

    print("Initial board:")
    print_board(env.get_state(), initial_board=puzzle)
    time.sleep(1)

    original_step = env.step
    def step_with_display(action):
        result = original_step(action)
        clear()
        print("=== AI Solving... ===\n")
        print_board(env.get_state(), initial_board=puzzle)
        time.sleep(0.3)
        return result
    env.step = step_with_display

    solved = solve_with_mrv(env, model)

    print()
    if solved:
        print("✓ Solved successfully!")
    else:
        print("✗ AI could not solve this puzzle.")

    input("\nPress Enter to return to the main menu...")


# ──────────────────────────────────────────────
#  Player Mode
# ──────────────────────────────────────────────
def run_player_mode():
    """
    Runs the player mode of the Sudoku game, allowing the user to solve a sudoku puzzle with
    a chosen difficulty level.

    Raised Errors:
        This function raises ValueError when inputs cannot be converted to integers.
e
    """
    clear()
    print("=== Player Mode ===\n")
    print("Select difficulty:")
    print("  1. Easy   (30 empty cells)")
    print("  2. Normal (40 empty cells)")
    print("  3. Hard   (50 empty cells)")

    diff_map = {"1": 30, "2": 40, "3": 50}
    while True:
        choice = input("Enter 1-3: ").strip()
        if choice in diff_map:
            empty_cells = diff_map[choice]
            break
        print("Invalid input. Please try again.")

    full_board = generate_sudoku(size=9)
    puzzle, solution = create_sudoku(full_board, empty_cells=empty_cells, size=9)
    env = SudokuEnv(puzzle, solution)
    env.reset()

    print("\nInput format:  row col digit   (1-indexed, e.g. 3 5 7)")
    print("Type 'u' to undo, 'q' to quit to main menu\n")

    while True:
        clear()
        print("=== Player Mode ===\n")
        state = env.get_state()
        print_board(state, initial_board=puzzle)

        empty_left = sum(1 for r in state for c in r if c == 0)
        print(f"\nCells remaining: {empty_left}")

        if env.is_solved():
            print("\n Congratulations! Puzzle complete!")
            input("Press Enter to return to the main menu...")
            return

        cmd = input("\nEnter (row col digit / u / q): ").strip().lower()

        if cmd == 'q':
            return

        if cmd == 'u':
            if env.undo():
                print("Last move undone.")
            else:
                print("Nothing to undo.")
            time.sleep(0.5)
            continue

        parts = cmd.split()
        if len(parts) != 3:
            input("Invalid format. Expected 'row col digit' (e.g. 3 5 7). Press Enter...")
            continue

        try:
            row, col, val = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
        except ValueError:
            input("Please enter numbers only. Press Enter...")
            continue

        if not (0 <= row <= 8 and 0 <= col <= 8 and 1 <= val <= 9):
            input("Row/col must be 1-9, digit must be 1-9. Press Enter...")
            continue

        if puzzle[row][col] != 0:
            input("That cell is part of the puzzle and cannot be changed. Press Enter...")
            continue

        state, _, done, _ = env.step((row, col, val))

        if state[row][col] != val:
            clear()
            print("=== Player Mode ===\n")
            print_board(env.get_state(), initial_board=puzzle, highlight=(row, col))
            print(f"\n✗ ({row+1}, {col+1}) = {val} is invalid — conflicts with row, column, or box.")
            input("Press Enter to continue...")


# ──────────────────────────────────────────────
#  Main Menu
# ──────────────────────────────────────────────
def main_menu():
    model = None  # lazy-load: only initialised when AI mode is selected

    while True:
        clear()
        print("╔══════════════════════════╗")
        print("║       Sudoku  Game       ║")
        print("╠══════════════════════════╣")
        print("║  1. Player Mode          ║")
        print("║  2. AI Solver Mode       ║")
        print("║  3. Quit                 ║")
        print("╚══════════════════════════╝")

        choice = input("\nSelect an option (1/2/3): ").strip()

        if choice == '1':
            run_player_mode()
        elif choice == '2':
            if model is None:
                print("\nLoading model...")
                model = load_model("cpu")
            run_ai_mode(model)
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            input("Invalid input. Press Enter to try again...")


if __name__ == '__main__':
    main_menu()
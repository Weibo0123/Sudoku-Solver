import time
import os
import torch
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from imitation.model import ImitationModel
from imitation.train import solve_with_mrv

def print_board(state):
    os.system('cls' if os.name == 'nt' else 'clear')
    print("+" + "---------+" * 3)
    for i, row in enumerate(state):
        line = "|"
        for j, val in enumerate(row):
            cell = str(val) if val != 0 else "."
            line += f" {cell} "
            if (j + 1) % 3 == 0:
                line += "|"
        print(line)
        if (i + 1) % 3 == 0:
            print("+" + "---------+" * 3)

def demo():
    model = ImitationModel()
    model.load_state_dict(torch.load('checkpoints/sudoku_model.pth'))
    model.eval()

    full_board = generate_sudoku(size=9)
    puzzle = create_sudoku(full_board, empty_cells=40, size=9)
    env = SudokuEnv(puzzle)
    env.reset()

    print_board(env.get_state())
    time.sleep(1)

    original_step = env.step
    def step_with_display(action):
        result = original_step(action)
        print_board(env.get_state())
        time.sleep(1)
        return result
    env.step = step_with_display

    solved = solve_with_mrv(env, model)
    print("Solved!" if solved else "Failed.")

if __name__ == '__main__':
    demo()
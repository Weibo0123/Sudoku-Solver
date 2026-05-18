# train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os

from imitation.dataset import generate_dataset
from imitation.model import ImitationModel
from env.sudoku_generator import generate_sudoku, create_sudoku
from env.sudoku_env import SudokuEnv


def solve_with_mrv(env, model):
    state = env.get_state()

    for _ in range(500):
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            if not env.undo():
                break
            state = env.get_state()
            continue

        cell_options = {}
        for r, c, v in valid_actions:
            if (r, c) not in cell_options:
                cell_options[(r, c)] = []
            cell_options[(r, c)].append(v)

        min_cell = min(cell_options, key=lambda k: len(cell_options[k]))
        options = cell_options[min_cell]

        if len(options) == 1:
            action = (min_cell[0], min_cell[1], options[0])
        else:
            x = torch.FloatTensor([c / 9.0 for row in state for c in row]).unsqueeze(0)
            cell_idx_tensor = torch.LongTensor([min_cell[0] * 9 + min_cell[1]])
            with torch.no_grad():
                _, out_num = model(x, cell_label=cell_idx_tensor)
            num_scores = out_num[0]
            best_num = max(options, key=lambda n: num_scores[n - 1].item())
            action = (min_cell[0], min_cell[1], best_num)

        state, _, done, _ = env.step(action)
        if done:
            break

    return env.is_solved()


def train():
    X, y_cell, y_num = generate_dataset(num_puzzles=500, size=9)
    X_tensor = torch.FloatTensor(X)
    y_cell_tensor = torch.LongTensor(y_cell)
    y_num_tensor  = torch.LongTensor(y_num)

    model = ImitationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    dataset    = TensorDataset(X_tensor, y_cell_tensor, y_num_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for X_batch, y_cell_batch, y_num_batch in dataloader:
            optimizer.zero_grad()
            out_cell, out_num = model(X_batch, cell_label=y_cell_batch)
            loss = loss_fn(out_cell, y_cell_batch) + loss_fn(out_num, y_num_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_cell, out_num = model(X_tensor, cell_label=y_cell_tensor)
                pred_cell = out_cell.argmax(dim=1)
                pred_num  = out_num.argmax(dim=1)
                acc_cell = (pred_cell == y_cell_tensor).float().mean().item()
                acc_num  = (pred_num  == y_num_tensor).float().mean().item()
                acc_both = ((pred_cell == y_cell_tensor) & (pred_num == y_num_tensor)).float().mean().item()
                avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Cell {acc_cell:.2%} | Num {acc_num:.2%} | Both {acc_both:.2%}")

    model.eval()
    solved, num_tests = 0, 100

    for _ in range(num_tests):
        full_board = generate_sudoku(size=9)
        puzzle     = create_sudoku(full_board, empty_cells=40, size=9)
        env        = SudokuEnv(puzzle)
        env.reset()

        if solve_with_mrv(env, model):
            solved += 1

    print(f"\nSuccess Rate: {solved}/{num_tests} ({solved / num_tests:.2%})")
    os.makedirs('../checkpoints', exist_ok=True)
    torch.save(model.state_dict(), '../checkpoints/sudoku_model.pth')

if __name__ == '__main__':
    train()
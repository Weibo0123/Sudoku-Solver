# train.py
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from imitation.dataset import generate_dataset
from imitation.model import ImitationModel
from env.sudoku_generator import generate_sudoku, create_sudoku
from env.sudoku_env import SudokuEnv


def train():
    X, y_cell, y_num = generate_dataset(num_puzzles=1000, size=9)
    X_tensor = torch.FloatTensor(X)
    y_cell_tensor = torch.LongTensor(y_cell)
    y_num_tensor = torch.LongTensor(y_num)

    model = ImitationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, y_cell_tensor, y_num_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


    for epoch in range(300):
        model.train()
        epoch_loss = 0
        for X_batch, y_cell_batch, y_num_batch in dataloader:
            optimizer.zero_grad()
            out_cell, out_num = model(X_batch)
            loss = loss_fn(out_cell, y_cell_batch) + loss_fn(out_num, y_num_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_cell, out_num = model(X_tensor)
                pred_cell = out_cell.argmax(dim=1)
                pred_num = out_num.argmax(dim=1)
                acc_cell = (pred_cell == y_cell_tensor).float().mean().item()
                acc_num = (pred_num == y_num_tensor).float().mean().item()
                acc_both = ((pred_cell == y_cell_tensor) & (pred_num == y_num_tensor)).float().mean().item()
                avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Cell {acc_cell:.2%} | Num {acc_num:.2%} | Both {acc_both:.2%}")

    model.eval()
    solved, num_tests = 0, 100

    for _ in range(num_tests):
        full_board = generate_sudoku(size=9)
        puzzle = create_sudoku(full_board, empty_cells=40, size=9)
        env = SudokuEnv(puzzle)
        state = env.reset()

        for _ in range(200):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            x = torch.FloatTensor([cell / 9.0 for row in state for cell in row]).unsqueeze(0)

            with torch.no_grad():
                out_cell, out_num = model(x)
            state_flat = [cell for row in state for cell in row]
            mask = torch.full((1, 81), float('-inf'))
            for idx, v in enumerate(state_flat):
                if v == 0:
                    mask[0, idx] = 0.0
            out_cell = out_cell + mask

            cell_idx = out_cell.argmax(dim=1).item()
            num = out_num.argmax(dim=1).item() + 1
            row, col = cell_idx // 9, cell_idx % 9
            action = (row, col, num)

            if action not in valid_actions:
                action = random.choice(valid_actions)

            state, _, done, _ = env.step(action)
            if done:
                break

        if env.is_solved():
            solved += 1

    print(f"\nSuccess Rate: {solved}/{num_tests} ({solved / num_tests:.2%})")

if __name__ == '__main__':
    train()
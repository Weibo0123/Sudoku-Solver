# train.py
from imitation.dataset import generate_dataset
from imitation.model import ImitationModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from env.sudoku_generator import generate_sudoku, create_sudoku
from env.sudoku_env import SudokuEnv
import random


def train():
    X, y = generate_dataset(num_puzzles=1000, size=9)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    model = ImitationModel(input_size=9*9, output_size=9*9*9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(500):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)

            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            output_all = model(X_tensor)
            predicted = output_all.argmax(dim=1)  # 0~728
            correct = (predicted == y_tensor).sum().item()
            accuracy = correct / len(y_tensor)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    full_board = generate_sudoku(size=9)
    puzzle = create_sudoku(full_board, empty_cells=40, size=9)
    env = SudokuEnv(puzzle)
    state = env.reset()

    solved = 0
    num_tests = 100

    for _ in range(num_tests):
        full_board = generate_sudoku(size=9)
        puzzle = create_sudoku(full_board, empty_cells=40, size=9)
        env = SudokuEnv(puzzle)
        state = env.reset()

        for _ in range(100):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            x = torch.FloatTensor([cell / 9.0 for row in state for cell in row]).unsqueeze(0)
            output = model(x)  # (1, 729)
            output = output.view(1, 81, 9)  # 推理时再 reshape

            # 空格掩码
            state_flat = [cell for row in state for cell in row]
            mask = torch.full((1, 81, 9), float('-inf'))
            for idx, v in enumerate(state_flat):
                if v == 0:
                    mask[0, idx, :] = 0
            output = output + mask

            best_scores = output[0].max(dim=1).values
            cell_idx = best_scores.argmax().item()
            num = output[0, cell_idx].argmax().item() + 1
            row, col = cell_idx // 9, cell_idx % 9

            action = (row, col, num)
            if action not in valid_actions:
                action = random.choice(valid_actions)

            state, reward, done, _ = env.step(action)
            if done:
                break

        if env.is_solved():
            solved += 1

if __name__ == '__main__':
    train()
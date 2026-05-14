# train.py
from imitation.dataset import generate_dataset
from imitation.model import ImitationModel
import torch
import torch.nn as nn
from env.sudoku_generator import generate_sudoku, create_sudoku
from env.sudoku_env import SudokuEnv
import random
from env.sudoku_generator import generate_sudoku, create_sudoku
from env.sudoku_env import SudokuEnv
import random


def train():
    X, y = generate_dataset(num_puzzles=1000, size=9)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    model = ImitationModel(input_size=9*9, output_size=9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            predicted = output.argmax(dim=1)
            correct = sum(1 for p, t in zip(predicted.tolist(), y_tensor.tolist()) if p == t)
            accuracy = correct / len(y_tensor)
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    train()
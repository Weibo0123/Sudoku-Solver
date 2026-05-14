# train.py
from imitation.dataset import generate_dataset
from imitation.model import ImitationModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train():
    X, y = generate_dataset(num_puzzles=1000, size=9)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    model = ImitationModel(input_size=9*9, output_size=9)
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
            predicted = output_all.argmax(dim=1)
            correct = sum(1 for p, t in zip(predicted.tolist(), y_tensor.tolist()) if p == t)
            accuracy = correct / len(y_tensor)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    train()
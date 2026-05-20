# Sudoku Solver

An AI-powered Sudoku solver experimenting with different machine learning methods, including Deep Reinforcement Learning (DQN) and Imitation Learning.


## Imitation Learning
Release

You can download the latest executable release here:

[v1.0.0 Release](https://github.com/Weibo0123/Sudoku-Solver/releases/tag/v1.0.0)

### How it works

- **Imitation Learning**: A CNN model is trained on expert backtracking solutions to predict which cell to fill and what number to place
- **MRV (Minimum Remaining Values)**: Prioritizes cells with the fewest legal options, directly filling cells with only one option without using the model
- **Backtracking**: If a dead end is reached, the solver undoes the last move and tries again

### Results

- 97% success rate on 40-cell puzzles

### Installation

Clone the repository:

```bash
git clone https://github.com/Weibo0123/Sudoku-Solver.git
cd Sudoku-Solver
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```
# train.py
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from dqn.dqn_agent import DQNAgent

STAGES = [
    {"size": 9, "empty_cells": 3,  "threshold": 0.8},
    {"size": 9, "empty_cells": 6,  "threshold": 0.8},
    {"size": 9, "empty_cells": 10, "threshold": 0.8},
    {"size": 9, "empty_cells": 15, "threshold": 0.8},
    {"size": 9, "empty_cells": 20, "threshold": 0.8},
    {"size": 9, "empty_cells": 40, "threshold": None},
]

def train():
    agent = DQNAgent(board_size=9, output_size=9*9*9)
    stage = 0
    solved = 0
    eval_interval = 100

    for episode in range(10000):
        current_stage = STAGES[stage]
        full_board = generate_sudoku(size=current_stage["size"])
        puzzle, solution = create_sudoku(full_board, empty_cells=current_stage["empty_cells"], size=current_stage["size"])
        env = SudokuEnv(puzzle, solution)
        state = env.reset()
        episode_losses = []
        done = False


        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break

            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            if env.is_solved():
                print(f"Episode {episode}: SOLVED!")
                solved += 1

        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            print(f"Episode {episode}: Avg Loss = {avg_loss:.4f}, epsilon: {agent.epsilon:.3f}")
        else:
            print(f"Episode {episode}: collecting experience, epsilon: {agent.epsilon:.3f}")

        if episode % eval_interval == 0 and episode > 0:
            success_rate = solved / eval_interval
            print(
                f"Episode {episode} | Stage {stage + 1} | empty_cells={current_stage['empty_cells']} | Success Rate = {success_rate:.2%}")
            solved = 0
            if current_stage["threshold"] and success_rate >= current_stage["threshold"]:
                stage += 1
                agent.epsilon = max(agent.epsilon, 0.3)
                print(f"Stage {stage + 1} threshold reached, epsilon: {agent.epsilon:.3f}")


if __name__ == '__main__':
    train()
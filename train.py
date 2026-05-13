# train.py
from sudoku_env import SudokuEnv
from dqn_agent import DQNAgent
from sudoku_generator import generate_sudoku, create_sudoku

def train():
    agent = DQNAgent(input_size=4*4, output_size=4*4*4)

    for episode in range(100):
        full_board = generate_sudoku(size=4)
        puzzle = create_sudoku(full_board, empty_cells=5, size=4)
        env = SudokuEnv(puzzle)
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

        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            print(f"Episode {episode}: Avg Loss = {avg_loss:.4f}, epsilon: {agent.epsilon:.3f}")
        else:
            print(f"Episode {episode}: collecting experience, epsilon: {agent.epsilon:.3f}")

if __name__ == '__main__':
    train()
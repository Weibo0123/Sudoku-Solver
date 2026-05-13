# train.py
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from dqn.dqn_agent import DQNAgent

def train():
    agent = DQNAgent(input_size=9*9, output_size=9*9*9)

    for episode in range(1000):
        full_board = generate_sudoku(size=9)
        puzzle = create_sudoku(full_board, empty_cells=20, size=9)
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
            if env.is_solved():
                print(f"Episode {episode}: SOLVED!")

        if episode_losses:
            avg_loss = sum(episode_losses) / len(episode_losses)
            print(f"Episode {episode}: Avg Loss = {avg_loss:.4f}, epsilon: {agent.epsilon:.3f}")
        else:
            print(f"Episode {episode}: collecting experience, epsilon: {agent.epsilon:.3f}")

if __name__ == '__main__':
    train()
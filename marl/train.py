import torch
import random
import os
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from marl.agent_manager import AgentManager
from marl.critic_network import CriticNetwork
from marl.arbitrator import Arbitrator
from marl.ppo_trainer import PPOTrainer

DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_EPISODES    = 10_000
EMPTY_CELLS     = 55       # number of cells to remove from full board
EVAL_EVERY      = 200      # evaluate every N episodes
EVAL_EPISODES   = 50       # number of episodes for evaluation
SAVE_EVERY      = 500      # save model every N episodes
SAVE_PATH       = "checkpoints/marl_sudoku.pt"

LR              = 1e-4
GAMMA           = 0.99
CLIP_PARAM      = 0.2
ENTROPY_COEF    = 0.1
VALUE_LOSS_COEF = 1.0
PPO_EPOCH       = 4


def evaluate(agent_manager, critic, arbitrator, num_episodes=50, empty_cells=55, device="cpu"):
    agent_manager.agent_network.eval()
    critic.eval()
    solved = 0

    for _ in range(num_episodes):
        full_board = generate_sudoku()
        puzzle, solution = create_sudoku(full_board, empty_cells=empty_cells)
        env = SudokuEnv(puzzle, solution)

        state = env.reset()
        agent_manager.reset_accuracy()
        arbitrator.reset(state)
        done = False

        while not done:
            board = env.get_state()
            empty = env.get_empty_cell()
            if not empty:
                break

            # MRV cell selection
            target_row, target_col = min(
                empty,
                key=lambda cell: sum(
                    env.is_valid(cell[0], cell[1], d) for d in range(1, 10)
                ),
            )

            with torch.no_grad():
                logits, _, _, _, mask_t = agent_manager.get_logits_with_grad(
                    env, board, target_row, target_col
                )

            if not mask_t[0].any():
                break

            action_mask = mask_t[0]
            scores = {
                "row": logits[0],
                "col": logits[1],
                "box": logits[2],
            }
            agent_indices = {
                "row": agent_manager.row_agent_idx(target_row),
                "col": agent_manager.col_agent_idx(target_col),
                "box": agent_manager.box_agent_idx(agent_manager.box_idx(target_row, target_col)),
            }

            digit = arbitrator.decide(scores, agent_indices, agent_manager, action_mask)
            _, _, done, _ = env.step((target_row, target_col, digit))

            correct = (digit == solution[target_row][target_col])
            for key in ("row", "col", "box"):
                agent_manager.update_accuracy(agent_indices[key], correct)
            arbitrator.update_remaining(agent_indices)

        if env.is_solved():
            solved += 1

    agent_manager.agent_network.train()
    critic.train()
    return solved / num_episodes


def save_checkpoint(agent_manager, critic, optimizer, episode, path):
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "episode": episode,
        "agent_network": agent_manager.agent_network.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(agent_manager, critic, optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    agent_manager.agent_network.load_state_dict(checkpoint["agent_network"])
    critic.load_state_dict(checkpoint["critic"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"  Loaded checkpoint from episode {checkpoint['episode']}")
    return checkpoint["episode"]


def train():
    print(f"Using device: {DEVICE}")

    # --- Initialise components ---
    agent_manager = AgentManager(device=DEVICE)
    critic = CriticNetwork().to(DEVICE)
    arbitrator = Arbitrator(device=DEVICE)
    trainer = PPOTrainer(
        agent_manager=agent_manager,
        critic=critic,
        arbitrator=arbitrator,
        device=DEVICE,
        lr=LR,
        gamma=GAMMA,
        clip_param=CLIP_PARAM,
        entropy_coef=ENTROPY_COEF,
        value_loss_coef=VALUE_LOSS_COEF,
        ppo_epoch=PPO_EPOCH,
    )

    best_solve_rate = 0.0

    for episode in range(1, NUM_EPISODES + 1):

        full_board = generate_sudoku()
        puzzle, solution = create_sudoku(full_board, empty_cells=EMPTY_CELLS)
        env = SudokuEnv(puzzle, solution)

        rollout = trainer.collect_rollouts(env, solution)

        losses = trainer.update(rollout)

        if episode % 50 == 0:
            print(
                f"Episode {episode:5d} | "
                f"policy_loss: {losses['policy_loss']:+.4f} | "
                f"value_loss: {losses['value_loss']:.4f} | "
                f"entropy: {losses['entropy_loss']:.4f}"
            )

        if episode % EVAL_EVERY == 0:
            solve_rate = evaluate(
                agent_manager, critic, arbitrator,
                num_episodes=EVAL_EPISODES,
                empty_cells=EMPTY_CELLS,
                device=DEVICE,
            )
            print(f"\n  Eval @ episode {episode}: solve rate = {solve_rate * 100:.1f}% | best = {best_solve_rate*100:.1f}%\n")

            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                save_checkpoint(
                    agent_manager, critic, trainer.optimizer, episode, SAVE_PATH
                )

        if episode % SAVE_EVERY == 0:
            save_checkpoint(
                agent_manager, critic, trainer.optimizer, episode,
                f"checkpoints/marl_sudoku_ep{episode}.pt"
            )

    print(f"\nTraining complete. Best solve rate: {best_solve_rate * 100:.1f}%")


if __name__ == "__main__":
    train()

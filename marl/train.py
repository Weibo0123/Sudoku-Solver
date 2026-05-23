import torch
import random
from env.sudoku_env import SudokuEnv
from env.sudoku_generator import generate_sudoku, create_sudoku
from marl.agent_manager import AgentManager
from marl.critic_network import CriticNetwork
from marl.arbitrator import Arbitrator
from marl.ppo_trainer import PPOTrainer

DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_EPISODES    = 10_000
EMPTY_CELLS     = 40       # number of cells to remove from full board
EVAL_EVERY      = 200      # evaluate every N episodes
EVAL_EPISODES   = 50       # number of episodes for evaluation
SAVE_EVERY      = 500      # save model every N episodes
SAVE_PATH       = "checkpoints/marl_sudoku.pt"

LR              = 3e-4
GAMMA           = 0.99
CLIP_PARAM      = 0.2
ENTROPY_COEF    = 0.01
VALUE_LOSS_COEF = 0.5
PPO_EPOCH       = 4


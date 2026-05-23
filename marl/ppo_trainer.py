# ppo_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from marl.agent_network import AGENT_ROW, AGENT_COL, AGENT_BOX

class PPOTrainer:
    def __init__(
            self,
            agent_manager,
            critic,
            arbitrator,
            device="cpu",
            lr: float = 3e-4,
            gamma: float = 0.99,
            clip_param: float = 0.2,
            entropy_coef: float = 0.01,
            value_loss_coef: float = 0.5,
            ppo_epoch: int = 4,
    ):
        self.agent_manager = agent_manager
        self.critic = critic
        self.arbitrator = arbitrator
        self.device = device

        self.gamma = gamma
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.ppo_epoch = ppo_epoch

        self.optimizer = optim.Adam(
            list(agent_manager.parameters()) + list(critic.parameterS()),
            lr=lr
        )

    def _select_cell_mrv(self, board, env):
        empty_cells = env.get_empty_cells()
        return min(
            empty_cells,
            key=lambda cell: sum(
                env.is_valid(cell[0], cell[1], d) for d in range(1, 10)
            ),
        )


    def collect_rollouts(self, env, solution):
        all_cell_values = []
        all_position = []
        all_agent_types = []
        all_mask = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_values = []
        all_boards = []

        state = env.reset()
        self.agent_manager.reset_accuracy()
        self.arbitrator.reset(state)
        done = False

        while not done:
            board = env.get_state()

            target_row, target_col = self._select_cell_mrv(board, env)

            board_flat = torch.tensor(
                [cell for row in board for cell in row], dtype=torch.long
            ).unsqueeze(0).to(self.device)
            value = self.critic(board_flat).squeeze()

            scores, agent_indices, action_mask = self.agent_manager.get_scores(
                env, board, target_row, target_col
            )

            row_cells, row_pos, col_cells, col_pos, box_cells, box_pos, box = self.agent_manager.get_actor_inputs(board, target_row, target_col)

            logits, cell_values_t, positions_t, agent_types_t, mask_t = self.agent_manager.get_logits_with_grad(env, board, target_row, target_col)

            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            digit = self.arbitrator.decide(scores, agent_indices, self.agent_manager, action_mask)

            digit_idx = digit - 1

            state, reward, done, _ = env.step((target_row, target_col, digit))

            correct = (digit == solution[target_row][target_col])
            for key in ("row", "col", "box"):
                self.agent_manager.reset_accuracy(agent_indices[key], correct)

            self.arbitrator.update_remaining(agent_indices)

            all_cell_values.append(cell_values_t)
            all_position.append(positions_t)
            all_agent_types.append(agent_types_t)
            all_mask.append(mask_t)
            all_actions.append(
                torch.tensor([digit_idx] * 3, dtype=torch.long).to(self.device)
            )
            all_log_probs.append(log_probs)
            all_rewards.append(reward)
            all_values.append(value.detach())
            all_boards.append(board_flat.squeeze(0))

        returns = self._compute_returns(all_rewards)

        return {
            "cell_values": torch.cat(all_cell_values),
            "positions": torch.cat(all_position),
            "agent_types": torch.cat(all_agent_types),
            "mask": torch.cat(all_mask),
            "actions": torch.cat(all_actions),
            "log_probs": torch.cat(all_log_probs),
            "rewards": torch.cat(all_rewards),
            "values": torch.cat(all_values),
            "boards": torch.cat(all_boards),
        }


    def _compute_returns(self, rewards: list) -> torch.Tensor:
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)


# ppo_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
            list(agent_manager.parameters()) + list(critic.parameters()),
            lr=lr
        )

    def _select_cell_mrv(self, env):
        empty_cells = env.get_empty_cell()
        return min(
            empty_cells,
            key=lambda cell: sum(
                env.is_valid(cell[0], cell[1], d) for d in range(1, 10)
            ),
        )


    def collect_rollouts(self, env, solution) -> dict:
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

            target_row, target_col = self._select_cell_mrv(env)

            board_flat = torch.tensor(
                [cell for row in board for cell in row], dtype=torch.long
            ).unsqueeze(0).to(self.device)
            value = self.critic(board_flat).squeeze()

            logits, cell_values_t, positions_t, agent_types_t, mask_t = self.agent_manager.get_logits_with_grad(env, board, target_row, target_col)

            action_mask = mask_t[0]
            scores = {"row": logits[0].detach(), "col": logits[1].detach(), "box": logits[2].detach()}
            agent_indices = {
                "row": self.agent_manager.row_agent_idx(target_row),
                "col": self.agent_manager.col_agent_idx(target_col),
                "box": self.agent_manager.box_agent_idx(self.agent_manager.box_idx(target_row, target_col)),
            }

            dist = Categorical(logits=logits)
            digit = self.arbitrator.decide(scores, agent_indices, self.agent_manager, action_mask)
            digit_idx = digit - 1
            digit_idx_t = torch.tensor([digit_idx] * 3, dtype=torch.long).to(self.device)
            log_probs = dist.log_prob(digit_idx_t)

            state, reward, done, _ = env.step((target_row, target_col, digit))

            correct = (digit == solution[target_row][target_col])
            for key in ("row", "col", "box"):
                self.agent_manager.update_accuracy(agent_indices[key], correct)

            self.arbitrator.update_remaining(agent_indices)

            all_cell_values.append(cell_values_t)
            all_position.append(positions_t)
            all_agent_types.append(agent_types_t)
            all_mask.append(mask_t)
            all_actions.append(
                torch.tensor([digit_idx] * 3, dtype=torch.long).to(self.device)
            )
            all_log_probs.append(log_probs.detach())
            all_rewards.append(reward)
            all_values.append(value.detach())
            all_boards.append(board_flat.squeeze(0))

        returns = self._compute_returns(all_rewards)

        return {
            "cell_values": torch.stack(all_cell_values),
            "positions": torch.stack(all_position),
            "agent_types": torch.stack(all_agent_types),
            "mask": torch.stack(all_mask),
            "actions": torch.stack(all_actions),
            "log_probs": torch.stack(all_log_probs),
            "returns": returns,
            "values": torch.stack(all_values),
            "boards": torch.stack(all_boards),
        }


    def _compute_returns(self, rewards: list) -> torch.Tensor:
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self, rollout: dict):
        cell_values = rollout["cell_values"]
        positions = rollout["positions"]
        agent_types = rollout["agent_types"]
        masks = rollout["mask"]
        actions = rollout["actions"]
        old_log_probs = rollout["log_probs"]
        returns = rollout["returns"]
        boards = rollout["boards"]

        T = cell_values.size(0)

        cv_flat = cell_values.view(T * 3, 9)
        pos_flat = positions.view(T * 3, 9)
        at_flat = agent_types.view(T * 3)
        mk_flat = masks.view(T * 3, 9)
        ac_flat = actions.view(T * 3)
        olp_flat = old_log_probs.view(T * 3)

        clip_loss = 0
        value_loss = 0
        entropy_loss = 0

        for _ in range(self.ppo_epoch):

            new_log_probs, entropy = self.agent_manager.agent_network.evaluate_actions(
                cv_flat, pos_flat, at_flat, mk_flat, ac_flat
            )

            new_values = self.critic(boards).squeeze()

            advantages = returns - new_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            advantages_flat = advantages.repeat_interleave(3)

            ratio = torch.exp(new_log_probs - olp_flat)
            clip_loss = -torch.min(
                ratio * advantages_flat,
                torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_flat,
            ).mean()

            value_loss = nn.functional.mse_loss(new_values, returns)

            entropy_loss = -entropy.mean()

            loss = clip_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.agent_manager.parameters()) + list(self.critic.parameters()),
                max_norm=0.5
            )
            self.optimizer.step()

        return {
            "policy_loss": clip_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

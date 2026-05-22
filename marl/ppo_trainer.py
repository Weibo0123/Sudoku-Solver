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
            list(agent_manager.parameters()) + list(critic.parameterS()),
            lr=lr
        )



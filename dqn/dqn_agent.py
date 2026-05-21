# dqn_agent.py
import torch.nn as nn
from dqn.q_network import QNetwork
from dqn.experience_replay import ReplayBuffer
import random
import torch
import copy

class DQNAgent:
    def __init__(self, board_size, output_size):
        self.board_size = board_size
        self.output_size = output_size
        self.q_network = QNetwork(board_size, output_size)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.0001)
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.target_network = copy.deepcopy(self.q_network)
        self.target_update_freq = 10
        self.train_steps = 0

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_tensor = self.state_to_tenser(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            valid_indices = [r * self.board_size * self.board_size + c * self.board_size + (v - 1) for r, c, v in valid_actions]
            valid_q = q_values.squeeze()[valid_indices]
            best_local = valid_q.argmax().item()
            return valid_actions[best_local]

    def state_to_tenser(self, state):
        size = self.board_size
        tensor = torch.zeros(size, size, size)
        for r in range (size):
            for c in range (size):
                v = state[r][c]
                if v > 0:
                    tensor[r][c][v - 1] = 1
        return tensor

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < 64:
            return None
        minibatch = self.replay_buffer.sample(64)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack([self.state_to_tenser(s) for s in states])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack([self.state_to_tenser(s) for s in next_states])
        dones = torch.FloatTensor(dones)
        action_indices = torch.LongTensor([
            r * self.board_size * self.board_size + c * self.board_size + (v - 1)
            for r, c, v in actions
        ])
        current_q_all = self.q_network(states)
        current_q = current_q_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q = torch.clamp(next_q, -1, 1)
            target_q = rewards + 0.99 * next_q.max(dim=1).values * (1 - dones)
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()



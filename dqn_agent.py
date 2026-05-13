# dqn_agent.py
from q_network import QNetwork
from experience_replay import ReplayBuffer
import random
import torch

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.q_network = QNetwork(input_size, output_size)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
            q_values = self.q_network(state_tensor)
            best_index = q_values.squeeze().argmax().item()
            best_index = min(best_index, len(valid_actions) - 1)
            return valid_actions[best_index]

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


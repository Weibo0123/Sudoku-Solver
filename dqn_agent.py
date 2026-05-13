# dqn_agent.py
from q_network import QNetwork
from experience_replay import ReplayBuffer

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.q_network = QNetwork(input_size, output_size)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

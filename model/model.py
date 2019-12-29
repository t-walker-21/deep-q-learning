"""
Class defintion for q learning agent
"""
import torch
import torch.nn as nn
import numpy as np
from .replay_buffer import ReplayBuffer

class DQNAgent(nn.Module):
    def __init__(self, state_space, action_space, eps=0.99, lr=1e-3, buf_len=1000, gamma=0.8):
        super(DQNAgent, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.eps = eps
        self.lr = lr
        self.replay_memory = ReplayBuffer(buf_len=buf_len)

        self.lin1 = nn.Linear(state_space, 10)
        self.lin2 = nn.Linear(10, action_space)
        self.gamma = gamma


    def store_experience(self, experience):
        """

        Store experience tuple in replay buffer
        """

        self.replay_memory.store_experience(experience)

    def forward(self, x):
        """

        Forward pass of DQN
        """

        x = torch.relu(self.lin1(x))
        x = self.lin2(x)

        return x


    def choose_action(self, state):
        """

        Select action according to q value or random
        """
        action = None

        if (self.eps > np.random.rand()): # Select random action if eps happened
            action = np.random.randint(self.action_space)

            #print ("selecting random action")

        else: # Get q_value with highest action
            q_values = self.forward(state)
            action = torch.argmax(q_values).item()

            #print ("selecting greedy action")

        if (self.eps > 0.1):
            self.eps *= 0.99

        return action
        

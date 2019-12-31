import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import time
import numpy as np

env = gym.make('CartPole-v0')

render = True

agent = DQNAgent(4, 2, buf_len=10, eps=0.0)
agent.load_state_dict(torch.load("checkpoints/cartpole/model_200.pt"))

# Alter environment
env.env.masscart *= 1
env.env.masspole *= 1e1

while True:

    state = env.reset()
    threshold = 0.25
    env.env.theta_threshold_radians = threshold
    longevity = 0
    
    while True:

        state_tensor = torch.Tensor(state)
        action = agent.choose_action(state_tensor)
        next_state, _, done, _ = env.step(action)
        done = int(done)

        longevity += 1

        if render:
            env.render()

        state = next_state
        if done:
            break
        
    print ("Episode len: ", longevity)

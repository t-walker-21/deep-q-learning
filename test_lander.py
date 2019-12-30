import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import time
import numpy as np

env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')

render = True

agent = DQNAgent(8, 4, buf_len=10, eps=0.01)
agent.load_state_dict(torch.load("checkpoints/lunarLander/model_0.pt"))

while True:

    state = env.reset()
    threshold = 0.25
    #env.env.theta_threshold_radians = threshold
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

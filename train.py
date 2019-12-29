import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import time
import numpy as np

env = gym.make('CartPole-v0')

render = True

agent = DQNAgent(4, 2, buf_len=10, eps=0.99)
target_net = DQNAgent(4, 2, buf_len=10)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 1e-2)
batch_size = 64

iteration_count = 0

while True:

    state =  env.reset()
    threshold = 0.3
    env.env.theta_threshold_radians = threshold

    if iteration_count % 50 == 0:
        target_net.load_state_dict(agent.state_dict())
        print (iteration_count)
        print ("updating target network")
    
    while True:

        state_tensor = torch.Tensor(state)
        action = agent.choose_action(state_tensor)

        observation = env.step(action)
        next_state, _, done, _ = observation
        done = int(done)
        reward = correct_rewards(observation, threshold / 2)


        action_tensor = torch.Tensor([action])
        next_state_tensor = torch.Tensor(next_state)
        reward_tensor = torch.Tensor(np.array([reward]))
        done_tensor = torch.Tensor(np.array([done]))


        experience = torch.cat([state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor])
        agent.store_experience(experience)


        if render:
            env.render()

        state = next_state

        if (len(agent.replay_memory.buffer) == agent.replay_memory.buf_len):
            learn(agent, target_net, opt, criterion, batch_size)

        if done:
            break

    iteration_count += 1

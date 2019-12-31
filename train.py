import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import time
import numpy as np
import threading


def get_user_input():
	while True:
		global thread_var
		thread_var = int(input())


env = gym.make('CartPole-v0')

render = False

agent = DQNAgent(4, 2, buf_len=10000, eps=0.1, decay=0.995)
target_net = DQNAgent(4, 2, buf_len=1000000)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 1e-3)
batch_size = 256

iteration_count = 0
global thread_var
thread_var = False

x = threading.Thread(target=get_user_input)
x.start()

while True:

    state =  env.reset()
    threshold = 0.35
    env.env.theta_threshold_radians = threshold

    if iteration_count % 50 == 0:
        target_net.load_state_dict(agent.state_dict())
        print (iteration_count)
        print ("updating target network")
        print ("espilon: ", agent.eps)

    if iteration_count % 200 == 0:
        print ("Saving model")
        torch.save(agent.state_dict(), "checkpoints/cartpole/model_" + str(iteration_count) + ".pt")

    longevity = 0
    
    while True:

        state_tensor = torch.Tensor(state)
        action = agent.choose_action(state_tensor)

        observation = env.step(action)
        next_state, _, done, _ = observation
        done = int(done)
        reward = correct_rewards(observation, threshold / 2)

        longevity += 1

        action_tensor = torch.Tensor([action])
        next_state_tensor = torch.Tensor(next_state)
        reward_tensor = torch.Tensor(np.array([reward]))
        done_tensor = torch.Tensor(np.array([done]))


        experience = torch.cat([state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor])
        agent.store_experience(experience)


        if thread_var:
            env.render()

        state = next_state

        if (len(agent.replay_memory.buffer) >= batch_size):
            learn(agent, target_net, opt, criterion, batch_size)

        if done:
            break

    iteration_count += 1
    agent.anneal_eps()

    print ("Episode len: ", longevity)

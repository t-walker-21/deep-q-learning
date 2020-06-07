import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn, correct_rewards_alt
import torch
import time
import numpy as np
import threading
import os


def get_user_input():
	while True:
		global thread_var
		thread_var = int(input())


env = gym.make('CartPole-v0')
checkpoint_dir = "checkpoints/cartpole/"

render = False
device = None


if torch.cuda.is_available():
    device = 'cuda'

else:
    device = 'cpu'


# agent = DQNAgent(4, 2, buf_len=10000, eps=0.7, decay=0.99).to(device)
agent = DQNAgent(4, 2, buf_len=40000, eps=0.9, decay=0.995).to(device)
target_net = DQNAgent(4, 2, buf_len=1000000).to(device)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 1e-3)
batch_size = 256

iteration_count = 0
global thread_var
thread_var = False

x = threading.Thread(target=get_user_input)
x.start()

# max_values = [np.NINF] * 4
threshold_array = np.array([2.4, 4.4, 0.4, 3.8]) / 2

# Create checkpoints folder if it doesn't exist.
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

while True:

    state = env.reset()
    threshold = 0.35
    env.env.theta_threshold_radians = threshold

    if iteration_count % 50 == 0:
        target_net.load_state_dict(agent.state_dict())
        print (iteration_count)
        print ("updating target network")
        print ("espilon: ", agent.eps)
        # print(max_values)

    if iteration_count % 20 == 0:
        print ("Saving model")
        torch.save(agent.state_dict(), checkpoint_dir + "model_" + str(iteration_count) + ".pt")

    longevity = 0
    
    while True:

        state_tensor = torch.Tensor(state).to(device)
        action = agent.choose_action(state_tensor)

        observation = env.step(action)
        next_state, _, done, _ = observation
        done = int(done)
        # reward = correct_rewards(observation, threshold / 2)
        reward = correct_rewards_alt(observation[0], threshold_array)

        longevity += 1

        action_tensor = torch.Tensor([action])
        next_state_tensor = torch.Tensor(next_state)
        reward_tensor = torch.Tensor(np.array([reward]))
        done_tensor = torch.Tensor(np.array([done]))

        for i, _ in enumerate(observation):
            if abs(observation[0][i]) > max_values[i]:
                max_values[i] = abs(observation[0][i])

        experience = torch.cat([state_tensor.cpu().detach(), action_tensor, reward_tensor, next_state_tensor, done_tensor])
        agent.store_experience(experience)


        if thread_var:
            env.render()

        state = next_state

        if (len(agent.replay_memory.buffer) >= batch_size):
            learn(agent, target_net, opt, criterion, batch_size, device)

        if done:
            break

    iteration_count += 1
    agent.anneal_eps()

    print ("Episode len: ", longevity)

import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import numpy as np
import threading

env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')

def thread_function():
    while True:
        global render
        render = int(input())

global render
render = False

print (env.observation_space)
print (env.action_space)


agent = DQNAgent(8, 4, buf_len=int(1e5), eps=0.99, decay=0.99995)
target_net = DQNAgent(8, 4, buf_len=1)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 1e-3)
batch_size = 256

iteration_count = 0
x = threading.Thread(target=thread_function)
x.start()


while iteration_count < 100000:

    state =  env.reset()


    if iteration_count % 10 == 0:
        target_net.load_state_dict(agent.state_dict())
        print (iteration_count)
        print ("updating target network")
        print ("epsilon: ", agent.eps)

    if iteration_count % 200 == 0:
        print ("Saving model")
        torch.save(agent, "checkpoints/lunarLander/model_" + str(iteration_count) + ".pt")

    longevity = []
    
    while True:

        state_tensor = torch.Tensor(state)
        action = agent.choose_action(state_tensor)

        observation = env.step(action)
        next_state, reward, done, _ = observation
        done = int(done)

        #reward = correct_rewards(observation, threshold / 2)

        longevity.append(reward)

        action_tensor = torch.Tensor([action])
        next_state_tensor = torch.Tensor(next_state)
        reward_tensor = torch.Tensor(np.array([reward]))
        done_tensor = torch.Tensor(np.array([done]))


        experience = torch.cat([state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor])
        agent.store_experience(experience)


        if render:
            env.render()

        state = next_state

        if (len(agent.replay_memory.buffer) >= batch_size):
            learn(agent, target_net, opt, criterion, batch_size)

        if done:
            break

    iteration_count += 1

    if iteration_count % 1 == 0:
        print ("Episode reward avg: ", np.mean(longevity))
        print ("Episode reward max: ", max(longevity))
        print ("Episode reward min: ", min(longevity))
        print ()

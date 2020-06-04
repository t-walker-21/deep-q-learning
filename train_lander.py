import gym
from model.model import DQNAgent
from utils.utils import correct_rewards, learn
import torch
import numpy as np
import threading
import os

env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
checkpoint_dir = "checkpoints/lunarLander/"

def thread_function():
    while True:
        global render
        render = int(input())

global render
render = False

device = None


if torch.cuda.is_available():
    device = 'cuda'

else:
    device = 'cpu'

agent = DQNAgent(8, 4, buf_len=int(1e6), eps=0.99, decay=0.995, gamma=0.99).to(device)
target_net = DQNAgent(8, 4, buf_len=1).to(device)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr = 5e-4)
batch_size = 64
save_rate = 100
update_rate = 20

iteration_count = 0
x = threading.Thread(target=thread_function)
x.start()

# Create checkpoints folder if it doesn't exist.
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

while True:

    state = env.reset()


    if iteration_count % update_rate == 0:
        target_net.load_state_dict(agent.state_dict())
        print ("Episode: ", iteration_count)
        print ("updating target network")
        print ("epsilon: ", agent.eps)

    if iteration_count % save_rate == 0:
        print ("Saving model")
        torch.save(agent.state_dict(), checkpoint_dir + "model_" + str(iteration_count) + ".pt")

    longevity = []
    
    while True:

        state_tensor = torch.Tensor(state).to(device)
        action = agent.choose_action(state_tensor)
        observation = env.step(action)
        next_state, reward, done, _ = observation
        #reward *= 1.0
        

        if (action > 0):
            reward -= 5

        done = int(done)

        #reward = correct_rewards(observation, threshold / 2)

        longevity.append(reward)

        action_tensor = torch.Tensor([action])
        next_state_tensor = torch.Tensor(next_state)
        reward_tensor = torch.Tensor(np.array([reward]))
        done_tensor = torch.Tensor(np.array([done]))


        experience = torch.cat([state_tensor.cpu().detach(), action_tensor, reward_tensor, next_state_tensor, done_tensor])
        agent.store_experience(experience)


        if render:
            env.render()

        state = next_state

        if (len(agent.replay_memory.buffer) >= batch_size):

            loss = learn(agent, target_net, opt, criterion, batch_size, device)

            if (iteration_count % 5 == 0):
                pass
                #print ("Loss: ", loss)

        if done:
            break

    iteration_count += 1

    agent.anneal_eps()

    if iteration_count % 1 == 0:
        print ("Episode reward avg: ", np.mean(longevity))
        print ("Episode reward max: ", max(longevity))
        print ("Episode reward min: ", min(longevity))
        print ()

"""

Utilities for dqn training
"""
import numpy as np
import torch
import cv2


def correct_rewards(tup, threshold, thres_tup=None):
    """

    I want my cartpole reward to be such that:
    if (pole angle < thr && pole_angle > -thr):
        reward = 1
    else:
        reward = 0
    """

    if abs(tup[0][2]) < threshold / 2:
        reward = 1

    else:
        reward = 0

    return reward


def correct_rewards_alt(tup, threshold_array):
    reward = 0
    for i, _ in enumerate(tup):
        reward += 1 - abs(tup[i]) / threshold_array[i]
    reward = reward / len(tup)
    return reward


def learn(agent, target, opt, criterion, batch_size, device):
    """

    Optimize agent 
    """

    # Get batch of data

    batch = agent.replay_memory.sample(batch_size)

    # Dissect batch
    # TODO: IMPLEMENT SLICING?
    state_batch = batch[0][:agent.state_space]
    action_batch = batch[0][agent.state_space].view(-1)
    reward_batch = batch[0][agent.state_space + 1].view(-1)
    next_state_batch = batch[0][agent.state_space + 2:agent.state_space * 2 + 2]
    done_batch = batch[0][-1].view(-1)

    for exp in batch:

        state_batch = torch.cat([state_batch, exp[:agent.state_space]])
        action_batch = torch.cat([action_batch, exp[agent.state_space].view(-1)])
        reward_batch = torch.cat([reward_batch, exp[agent.state_space + 1].view(-1)])
        next_state_batch = torch.cat([next_state_batch, exp[agent.state_space + 2:agent.state_space * 2 + 2]])
        done_batch = torch.cat([done_batch, exp[-1].view(-1)])

    state_batch = state_batch.view(batch_size + 1, -1)[1:].to(device)
    action_batch = action_batch.view(batch_size + 1, -1)[1:].long().to(device)
    reward_batch = reward_batch.view(batch_size + 1, -1)[1:].to(device)
    next_state_batch = next_state_batch.view(batch_size + 1, -1)[1:].to(device)
    done_batch = done_batch.view(batch_size + 1, -1)[1:].to(device)

    # Get Q-values

    q_values = agent(state_batch).gather(1, action_batch)

    # Get target values

    mask = 1 - done_batch

    target_val = torch.max(target(next_state_batch), dim=1)[0].view(batch_size, -1)
    targets = reward_batch + mask * (agent.gamma * target_val)

    loss = criterion(q_values, targets)

    #print (loss)

    opt.zero_grad()

    loss.backward()

    opt.step()
    
    return loss

def learn_image(agent, target, opt, criterion, batch_size, device, image_shape=(100,100)):
    """

    Optimize agent 
    """

    # Get batch of data

    batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.replay_memory.sample(batch_size)

    # Dissect batch

    batch_state = torch.tensor(batch_state).view(batch_size, 1, image_shape[0], image_shape[1]).to(device).float()
    batch_next_state = torch.tensor(batch_next_state).view(batch_size, 1, 100, 100).to(device).float()
    batch_action = torch.tensor(batch_action).to(device).long().view(batch_size, 1)
    batch_reward = torch.tensor(batch_reward).to(device).float()
    batch_done = torch.tensor(batch_done).to(device).float()

    # Get Q-values

    q_values = agent(batch_state).gather(1, batch_action)

    # Get target values

    mask = 1 - batch_done

    target_val = torch.max(target(batch_next_state), dim=1)[0].view(batch_size, -1)
    targets = batch_reward + mask * (agent.gamma * target_val)

    loss = criterion(q_values, targets)

    #print ("loss")
    print (loss.item())

    opt.zero_grad()

    loss.backward()

    opt.step()
    
    return loss
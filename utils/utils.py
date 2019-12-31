"""

Utilities for dqn training
"""
import numpy as np
import torch

def correct_rewards(tup, threshold):
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

def learn(agent, target, opt, criterion, batch_size):
    """

    Optimize agent 
    """

    # Get batch of data

    batch = agent.replay_memory.sample(batch_size)

    # Dissect batch

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

    state_batch = state_batch.view(batch_size + 1, -1)[1:]
    action_batch = action_batch.view(batch_size + 1, -1)[1:].long()
    reward_batch = reward_batch.view(batch_size + 1, -1)[1:]
    next_state_batch = next_state_batch.view(batch_size + 1, -1)[1:]
    done_batch = done_batch.view(batch_size + 1, -1)[1:]

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
    
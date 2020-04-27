from model.model import PGAgent
import torch
import torch.optim as optim
import gym
from utils.utils import correct_rewards
import numpy as np

render = True

if torch.cuda.is_available():
	device = 'cuda' 

else: 
	device = 'cpu'


# Instantiate agent and environment

env = gym.make('LunarLander-v2')

agent = PGAgent(8, 4, 32).to(device)

GAMMA = 0.9

# Create optimizer

opt = optim.Adam(agent.parameters(), lr=5e-3)


def reinforce(agent):
	"""

	Increase prob of good actions and decrease prob of bad actions
	"""

	# Normalize discounted rewards (baseline)

	discounted_rewards = []

	for t in range(len(agent.rewards)):
		Gt = 0
		pwr = 0

		for r in agent.rewards[t:]:
			Gt += r * GAMMA**pwr
			pwr += 1

		discounted_rewards.append(Gt)

	norm_rewards = []

	mean = np.mean(discounted_rewards)
	std = np.std(discounted_rewards)

	for r in discounted_rewards:
		norm_rewards.append((r - mean) / std)


	# Calculate loss

	loss = torch.zeros(1).to(device)

	t = 0
	for G_t, log_prob in zip(norm_rewards, agent.log_probs):
		#print (G_t, log_prob)
		loss = loss + (G_t * log_prob * -1)
		t += 1

	opt.zero_grad()
	loss.backward()
	opt.step()

episode_count = 0

# Train loop
while True:

	# Initial state
	state = env.reset()
	agent.flush_buffers()
	ep_duration = 0
	reward_acc = []

	# Play through an episode
	while True:

		if (render):
			env.render()

		state_tensor = torch.Tensor(state).to(device)
		action_tensor = agent.select_action(state_tensor)
		action = action_tensor.item()
		observation = env.step(action)

		next_state, reward, done, _ = observation

		#reward = correct_rewards(observation, threshold / 2)

		agent.rewards.append(reward)

		reward_acc.append(reward)

		state = next_state

		ep_duration += 1

		if (done): # Reinforce and repeat
			reinforce(agent)
			print ("Episode " , episode_count)
			print ("Average reward: ", np.mean(reward_acc))
			print ("Min reward: ", np.min(reward_acc))
			print ("Max reward: ", np.max(reward_acc))
			break

	episode_count += 1
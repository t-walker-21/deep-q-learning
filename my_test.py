import gym
import numpy as np
import torch

# env = gym.make('CartPole-v0')
# env.reset()
# for _ in range(200):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


my_array = np.zeros((3, 3))
my_array[2, 2] = 1
print(my_array)
my_array = np.zeros_like(my_array)
print(my_array)

first_list = [2, 3, 4]
combined_list = [200] + first_list
print(combined_list)

print((3, 2, 5) + (100,))

my_tensor = torch.Tensor(np.arange(9*3)).view(3, 1, 3, 3)
top_row = my_tensor[:, :, 0, :]
top_row = top_row.view(3, -1)
print(my_tensor)
my_tensor = my_tensor.view(3, -1)
my_concat = torch.cat((my_tensor, top_row), dim=1)
print(my_concat)



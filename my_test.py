import gym
import numpy as np

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

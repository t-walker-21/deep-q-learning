"""

Class definition for replay buffer
"""
import random
import collections
import numpy as np

class ReplayBuffer(object):
	def __init__(self, buf_len=1000):
		"""

		Initialize replay buffer 
		"""
		super(ReplayBuffer, self).__init__()

		self.buf_len = buf_len
		self.buffer = collections.deque(maxlen=buf_len)

	def store_experience(self, experience):
		"""

		Take in experience tuple and store in deque
		"""

		self.buffer.append(experience)

	def sample(self, batch_size):
		"""

		Sample a random batch of tuples from memory and return
		"""

		batch = random.choices(self.buffer, k=batch_size)

		return batch

	def clear_buffer(self):
		"""

		Empty replay buffer
		"""

		self.buffer = collections.deque(maxlen=self.buf_len)

class ReplayBufferMulti(object):
	"""

	This buffer holds each item of RL tuple (sarsa) in multiple buffers because the items are of different dimensions
	"""
	def __init__(self, state_size, buf_len=1000):
		"""

		Initialize replay buffer 
		"""
		super(ReplayBufferMulti, self).__init__()

		state_size_overall = (buf_len, state_size[0], state_size[1])

		self.buf_len = buf_len
		self.ptr = 0
		self.buffer_state = np.zeros(state_size_overall)
		self.buffer_action = np.zeros(buf_len)
		self.buffer_reward = np.zeros(buf_len)
		self.buffer_next_state = np.zeros(state_size_overall)
		self.buffer_done = np.zeros(buf_len)
		self.state_size=state_size
		self.ready = False

	def store_experience(self, experience):
		"""

		Take in experience tuple and store in deque
		"""

		self.buffer_state[self.ptr % self.buf_len] = experience[0]
		self.buffer_action[self.ptr % self.buf_len] = experience[1]
		self.buffer_reward[self.ptr % self.buf_len] = experience[2]
		self.buffer_next_state[self.ptr % self.buf_len] = experience[3]
		self.buffer_done[self.ptr % self.buf_len] = experience[4]

		self.ptr += 1

		if self.ptr == self.buf_len:
			self.ptr = 0
			self.ready = True

	def sample(self, batch_size):
		"""

		Sample a random batch of tuples from memory and return
		"""

		rand = np.random.randint(0, self.buf_len, batch_size)

		batch_state = self.buffer_state[rand]
		batch_action = self.buffer_action[rand]
		batch_reward = self.buffer_reward[rand]
		batch_next_state = self.buffer_next_state[rand]
		batch_done = self.buffer_done[rand]

		return batch_state, batch_action, batch_reward, batch_next_state, batch_done

	def clear_buffer(self):
		"""

		Empty replay buffer
		"""

		self.ptr = 0
		self.buffer_state = np.zeros_like(self.state_size, buf_len)
		self.buffer_action = np.zeros(buf_len)
		self.buffer_reward = np.zeros(buf_len)
		self.buffer_next_state = np.zeros_like(self.state_size, buf_len)
		self.buffer_done = np.zeros(buf_len)

		self.ready = False

	def is_ready(self):
		return self.ready
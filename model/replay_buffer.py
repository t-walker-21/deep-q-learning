"""

Class definition for replay buffer
"""
import random
import collections

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
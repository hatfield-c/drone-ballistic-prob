import torch

import CONFIG

class DataLoader:
	def __init__(self):
		self.data = torch.load(CONFIG.data_path)
		self.data = self.data[self.data[:, -1] > 0.01]
		self.data[:, -1] = 1
		
		print("[data shape]:", self.data.shape)
		
	def DrawSamples(self, batch_size):
		
		indices = torch.randint(0, self.data.shape[0], (batch_size,))
		
		batch = self.data[indices].cuda()
		samples = batch[:, :-1]
		values = batch[:, [-1]]
		
		if not CONFIG.is_radial:
			values = batch[:, [3, 4, 5]]
		
		return samples, values
	
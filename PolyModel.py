import torch
import copy
import numpy as np

import CONFIG

class PolyModel(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.dimensionality = CONFIG.input_size

		self.h0 = torch.nn.Linear(self.dimensionality, CONFIG.h_count).cuda()
		self.h1 = torch.nn.Linear(CONFIG.h_count, CONFIG.h_count).cuda()
		self.h2 = torch.nn.Linear(CONFIG.h_count, CONFIG.h_count // 8).cuda()
		self.final = torch.nn.Linear(CONFIG.h_count // 8, CONFIG.output_size).cuda()
		
		self.relu = torch.nn.ReLU()
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, data):
		
		out = self.h0(data)
		out = self.relu(out)
		out = self.h1(out)
		out = self.relu(out)
		out = self.h2(out)
		out = self.relu(out)
		out = self.final(out)
		#out = self.relu(out)
		#out = self.sigmoid(out)
		
		return out
	
	
	def Save(self, epoch):

		filename = "polyfield_" + str(epoch) + ".pt"
		save_path = CONFIG.model_path# + filename

		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, epoch):
		filename = "polyfield_" + str(epoch) + ".pt"
		save_path = CONFIG.model_path# + filename

		self.load_state_dict(torch.load(save_path))
		self.eval()

		print("\nLoaded model from", save_path)

	
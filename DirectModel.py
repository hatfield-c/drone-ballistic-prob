import torch
import copy
import numpy as np

import CONFIG

class DirectModel(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.dimensionality = CONFIG.input_size - 3

		self.h0 = torch.nn.Linear(self.dimensionality, CONFIG.h_count).cuda()
		self.h1 = torch.nn.Linear(CONFIG.h_count, CONFIG.h_count).cuda()
		self.h2 = torch.nn.Linear(CONFIG.h_count, CONFIG.h_count // 8).cuda()
		self.final = torch.nn.Linear(CONFIG.h_count // 8, CONFIG.output_size).cuda()
		
		self.relu = torch.nn.ReLU()
		self.sigs = torch.FloatTensor([[0.5, 3, 0.2]]).cuda()
		self.sigs = torch.square(self.sigs)

	def forward(self, data):
		
		in_data = data[:, [0, 1, 2, 6, 7, 8, 9, 10, 11]]
		u_data = data[:, 3:6]
		out = self.Inference(in_data)
		
		return out

	def Inference(self, data):	
		
		out = self.h0(data)
		out = self.relu(out)
		out = self.h1(out)
		out = self.relu(out)
		out = self.h2(out)
		out = self.relu(out)
		out = self.final(out)
		
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

	
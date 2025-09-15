import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

import CONFIG
import RadialModel
import DirectModel

class PlotMetrics:
	def __init__(self):
		self.relu = torch.nn.ReLU()
	
	def Plot(self):
		data = torch.load(CONFIG.verify_data_path).cuda()
		print("data shape:", data.shape)
		
		x = [1000, 10000, 50000]
		y_direct_suc = []
		y_direct_cep = []
		y_radial_suc = []
		y_radial_cep = []
		
		for epochs in x:
			model = DirectModel.DirectModel()
			model_path = "data/models/direct_" + str(epochs) + ".pt" 
			metrics = self.GetMetrics(data, model, model_path)
			
			y_direct_suc.append(metrics["average_success"])
			y_direct_cep.append(metrics["cep"])
			
			model = RadialModel.RadialModel()
			model_path = "data/models/radial_" + str(epochs) + ".pt" 
			metrics = self.GetMetrics(data, model, model_path)
			
			y_radial_suc.append(metrics["average_success"])
			y_radial_cep.append(metrics["cep"])
		
		fig = plt.figure()
		
		ax0 = plt.subplot(1, 2, 1)
		ax1 = plt.subplot(1, 2, 2)
		
		ax0.plot(x, y_direct_cep, c = "g")
		ax0.scatter(x, y_direct_cep, c = "g")
		ax0.plot(x, y_radial_cep, c = "b")
		ax0.scatter(x, y_radial_cep, c = "b")
		
		ax1.plot(x, y_direct_suc, c = "g")
		ax1.scatter(x, y_direct_suc, c = "g")
		ax1.plot(x, y_radial_suc, c = "b")
		ax1.scatter(x, y_radial_suc, c = "b")
		
		plt.show()
		
	def GetMetrics(self, init_states, model, model_path):
		model.Load(model_path)
		model = model.cuda()
		
		### REMOVE
		init_states = init_states[:1000]
		###
		
		p0_states = init_states[:, :3].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
		v0_states = torch.zeros(p0_states.shape).cuda()
		pv_states = torch.cat((p0_states, v0_states), dim = 2)
		
		mus = init_states[:, 6:9].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
		sigmas = init_states[:, 9:12].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
		
		in_data = init_states[:, [0, 1, 2, 6, 7, 8, 9, 10, 11]]
		us = model.Inference(in_data).detach().reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
		#us = init_states[:, 3:6].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
		
		gravity = torch.zeros((pv_states.shape[0], 1, 3)).cuda()
		gravity[:, 0, 1] = -9.8
		rad_p = 1
		
		hits = torch.zeros((pv_states.shape[0], CONFIG.wind_samples)).cuda()
		w_signal = torch.ones((pv_states.shape[0], CONFIG.wind_samples)).cuda()
		smallest_distances = torch.ones((pv_states.shape[0], CONFIG.wind_samples)).cuda() * 1000
		cep = torch.zeros((pv_states.shape[0], CONFIG.wind_samples)).cuda()
		
		for i in range(CONFIG.sim_steps):
			
			thrust = torch.zeros(1, 1, 3).cuda()
			if i < CONFIG.flight_steps:
				thrust = us
			
			wind = torch.normal(mus, sigmas)
			acceleration = gravity + wind + thrust

			pv_states[:, :, :3] += pv_states[:, :, 3:6] * CONFIG.delta_time
			pv_states[:, :, 3:6] += acceleration * CONFIG.delta_time
			
			# wall collision with padding
			w0_signal = pv_states[:, :, 1]
			w1_signal = pv_states[:, :, 2]
			w0_signal = self.relu(torch.abs(w0_signal - 3) - 3)
			w1_signal = self.relu(torch.abs(w1_signal - (-5)) - 1.5)
			w0_signal = torch.ceil(w0_signal / 1000)
			w1_signal = torch.ceil(w1_signal / 1000)
			
			ws = torch.maximum(w0_signal, w1_signal)
			w_signal = torch.minimum(w_signal, ws)
			
			p_signal = hits
			if i >= CONFIG.flight_steps:
				p_signal = pv_states[:, :, :3]
				distances = torch.linalg.norm(p_signal, dim = 2)
				
				p_signal = rad_p - distances
				p_signal = self.relu(p_signal) / 1000
				p_signal = torch.ceil(p_signal)
				
				smallest_distances = torch.minimum(smallest_distances, distances)
			
			hit_signal = p_signal	
			
			hits = torch.maximum(hit_signal, hits)
			hits = torch.minimum(w_signal, hits)
			
		average_success = torch.mean(hits)
		average_closest_distance = torch.mean(smallest_distances)
		
		sorted_distances = torch.sort(smallest_distances, dim = 1).values
		cep = sorted_distances[:, sorted_distances.shape[1] // 2]
		cep = torch.mean(cep)
		
		results = { "average_success": average_success.cpu().numpy(), "closest_distance": average_closest_distance.cpu().numpy(), "cep": cep.cpu().numpy() }
		
		return results
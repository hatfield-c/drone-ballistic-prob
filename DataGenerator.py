import torch
import numpy as np
import matplotlib.pyplot as plt

import CONFIG

class DataGenerator:
	def __init__(self):
		self.relu = torch.nn.ReLU()
	
	def Generate(self):
		data = torch.zeros(1, CONFIG.low.shape[0] + 1)
		
		print("[Ballistic Thrust Probability Generator]")
		print("    Delta time:     :", CONFIG.delta_time)
		print("    Flight steps    :", CONFIG.flight_steps)
		print("    Ballistic steps :", CONFIG.ballistic_steps)
		print("")
		
		print_count  = 0
		
		while data.shape[0] < CONFIG.sample_count:
			init_states = torch.rand((CONFIG.sim_batch_size, CONFIG.low.shape[0])).cuda()
			init_states = init_states * CONFIG.width.reshape(1, -1)
			init_states += CONFIG.low.reshape(1, -1)
			
			p0_states = init_states[:, :3].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			v0_states = torch.zeros(p0_states.shape).cuda()
			pv_states = torch.cat((p0_states, v0_states), dim = 2)
			
			us = init_states[:, 3:6].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			
			mus = init_states[:, 6:9].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			sigmas = init_states[:, 9:12].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			
			gravity = torch.zeros((pv_states.shape[0], 1, 3)).cuda()
			gravity[:, 0, 1] = -9.8
			rad_p = 1
			
			hits = torch.zeros((pv_states.shape[0], CONFIG.wind_samples)).cuda()
			w_signal = torch.ones((pv_states.shape[0], CONFIG.wind_samples)).cuda()
			
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
					p_signal = rad_p - torch.linalg.norm(p_signal, dim = 2)
					p_signal = self.relu(p_signal) / 1000
					p_signal = torch.ceil(p_signal)
				
				hit_signal = p_signal	
				
				hits = torch.maximum(hit_signal, hits)
				hits = torch.minimum(w_signal, hits)
				
			probs = torch.sum(hits, dim = 1) / CONFIG.wind_samples
			probs = probs.reshape(-1, 1)
			
			if print_count >= 10:
				print("Samples:", data.shape[0])
				print_count = 0
			print_count += 1
				
			#print("    Hits:", torch.sum(torch.max(hits, dim = 1).values).cpu().item())
			batch_data = torch.cat((init_states, probs), dim = 1).cpu()
			batch_data = batch_data[batch_data[:, -1] > 0]
			
			if batch_data.shape[0] < 1:
				continue
			
			data = torch.cat((data, batch_data))

		print("    Final shape:", data.shape)
		print("Saving...")
		torch.save(data, CONFIG.data_path)
		
		print("    Done!")
		
	
		
		
import torch
import numpy as np
import matplotlib.pyplot as plt

import CONFIG

class BezierGenerator:
	def __init__(self):
		self.relu = torch.nn.ReLU()
	
	def Generate(self, batch_size):
		data = []
		
		thrust_memory = None
		p_memory = None
		v_memory = None
		
		print("[Bezier Thrust Probability Generator]")
		print("    Delta time:     :", CONFIG.delta_time)
		print("    Flight steps    :", CONFIG.flight_steps)
		print("    Ballistic steps :", CONFIG.ballistic_steps)
		print("    Sim steps       :", CONFIG.sim_steps)
		print("    Sim batch count :", CONFIG.sim_batch_count)
		print("    Sim batch size  :", CONFIG.sample_count)
		print("    Total samples   :", CONFIG.total_samples)
		print("")
		
		for i in range(CONFIG.sim_batch_count):
			print("Batch", i)
			print("    Building init states...")
			init_states = torch.rand((batch_size, CONFIG.low.shape[0])).cuda()
			init_states = init_states * CONFIG.width.reshape(1, -1)
			init_states += CONFIG.low.reshape(1, -1)
			
			p0_states = init_states[:, :3].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			v0_states = torch.zeros(p0_states.shape).cuda()
			pv_states = torch.cat((p0_states, v0_states), dim = 2)
			
			k1s = init_states[:, 3:6].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			k2s = init_states[:, 6:9].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			k3s = init_states[:, 9:12].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			k0s = torch.zeros(k1s.shape).cuda()
			k0s[:, :, 1] = 10
			
			mus = init_states[:, 12:15].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			sigmas = init_states[:, 15:18].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			
			gravity = torch.zeros((pv_states.shape[0], 1, 3)).cuda()
			gravity[:, 0, 1] = -9.8
			rad_p = 1
			rad_v = 0.7
			end_v = torch.FloatTensor([0, -1, 1])
			end_v = end_v / torch.linalg.norm(end_v)
			end_v = end_v.reshape(1, 1, 3).cuda()
			
			hits = torch.zeros((pv_states.shape[0], CONFIG.wind_samples)).cuda()
			
			print("    Simulating...")
			print("        ", end = "")
			
			thrust_memory = []
			p_memory = []
			v_memory = []
			a_memory = []
			for i in range(CONFIG.sim_steps):
				if i % (CONFIG.sim_steps // 10) == 0:
					print(str(i) + ", ", end = "")
				
				thrust = torch.zeros(1, 1, 3).cuda()
				if i < CONFIG.flight_steps:
					t = i / (CONFIG.flight_steps - 1)
					thrust = (((1 - t) ** 3) * k0s) + (t * k1s * (3 * ((1 - t) ** 2))) + (k2s * (3 * (1 - t) * (t ** 2))) + (k3s * (t ** 3))
				
				wind = torch.normal(mus, sigmas)
				acceleration = gravity + wind + thrust
	
				pv_states[:, :, :3] += pv_states[:, :, 3:6] * CONFIG.delta_time
				pv_states[:, :, 3:6] += acceleration * CONFIG.delta_time
				
				thrust_memory.append(thrust[0, 0])
				p_memory.append(pv_states[0, 0, :3].clone())
				v_memory.append(pv_states[0, 0, 3:6].clone())
				a_memory.append(acceleration[0, 0])
				
				# todo: don't count a hit if drone falls below 2 meters or if drone comes within 5 meters of target
				# todo: only save positive samples
				
				s_signal = hits
				if i >= CONFIG.flight_steps:
					p_signal = pv_states[:, :, :3]
					p_signal = rad_p - torch.linalg.norm(p_signal, dim = 2)
					p_signal = self.relu(p_signal) / 1000
					p_signal = torch.ceil(p_signal)
					
					v_signal = pv_states[:, :, 3:6]
					v_signal = v_signal / (torch.linalg.norm(v_signal, dim = 2, keepdim = True) + 0.00000001)
					v_signal = v_signal * end_v
					v_signal = torch.sum(v_signal, dim = 2)
					v_signal = self.relu(v_signal)
					v_signal = v_signal - rad_v
					v_signal = self.relu(v_signal) / 1000
					v_signal = torch.ceil(v_signal)
					
					s_signal = p_signal# * v_signal
				
				hits = torch.maximum(s_signal, hits)
				
			probs = torch.sum(hits, dim = 1) / CONFIG.wind_samples
			probs = probs.reshape(-1, 1)
			
			print("\n    Hits:", torch.sum(torch.max(hits, dim = 1).values).cpu().item())
			print("    Compiling batch...")
			batch_data = torch.cat((init_states, probs), dim = 1).cpu()
			data.append(batch_data)
			
		print("Compiling all...")
		data = torch.cat(data, dim = 0)
		print("    Final shape:", data.shape)

		print("Saving...")
		torch.save(data, CONFIG.data_path)
		
		print("    Done!")
		
	
		
		
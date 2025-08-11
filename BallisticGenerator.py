import torch

import CONFIG

class BallisticGenerator:
	def __init__(self):
		self.relu = torch.nn.ReLU()
	
	def Generate(self, batch_size):
		data = []
		
		for i in range(CONFIG.sim_batch_count):
			print("Batch", i)
			print("    Building init states...")
			init_states = torch.rand((batch_size, CONFIG.low.shape[0])).cuda()
			init_states = init_states * CONFIG.width.reshape(1, -1)
			init_states += CONFIG.low.reshape(1, -1)
			
			pv_states = init_states[:, :6].reshape(-1, 1, 6).repeat(1, CONFIG.wind_samples, 1)
			mus = init_states[:, 6:9].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			sigmas = init_states[:, 9:12].reshape(-1, 1, 3).repeat(1, CONFIG.wind_samples, 1)
			radii = init_states[:, 12].reshape(-1, 1)
			
			gravity = torch.zeros((pv_states.shape[0], 1, 3)).cuda()
			gravity[:, 0, 1] = -9.8
			
			hits = torch.zeros((pv_states.shape[0], CONFIG.wind_samples)).cuda()
			
			print("    Simulating...")
			print("        ", end = "")
			
			for i in range(CONFIG.sim_steps):
				if i % (CONFIG.sim_steps // 4) == 0:
					print(str(i) + ", ", end = "")
				
				wind = torch.normal(mus, sigmas)
				acceleration = gravity + wind
	
				pv_states[:, :, :3] += pv_states[:, :, 3:6] * CONFIG.delta_time
				pv_states[:, :, 3:6] += acceleration * CONFIG.delta_time
			
				d_signal = pv_states[:, :, :3]
				d_signal = radii - torch.linalg.norm(d_signal, dim = 2)
				d_signal = self.relu(d_signal) / 1000
				d_signal = torch.ceil(d_signal)
				
				hits = torch.maximum(d_signal, hits)
				
			probs = torch.sum(hits, dim = 1) / CONFIG.wind_samples
			probs = probs.reshape(-1, 1)
			
			print("\n    Hits:", torch.sum(hits).cpu().item())
			print("    Compiling batch...")
			batch_data = torch.cat((init_states, probs), dim = 1).cpu()
			data.append(batch_data)
		
		print("Compiling all...")
		data = torch.cat(data, dim = 0)
		print("    Final shape:", data.shape)

		print("Saving...")
		torch.save(data, CONFIG.data_path)
		
		print("    Done!")
		
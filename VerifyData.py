import torch
import numpy as np
import matplotlib.pyplot as plt

import CONFIG

class VerifyData:
	def __init__(self):
		pass
	
	def Verify(self):
		data = torch.load(CONFIG.data_path)
		print(data.shape)
		data = data[data[:, -1] > 0.5]
		print(data.shape)
		
		indices = torch.argsort(data[:, -1])
		
		for k in range(1, indices.shape[0] - 1):
			verify_index = indices[-k]
			
			x = data[verify_index]
			
			p0_state = x[:3]
			v0_state = torch.zeros(p0_state.shape)
			pv_state = torch.cat((p0_state, v0_state))
			
			k1 = x[3:6]
			k2 = x[6:9]
			k3 = x[9:12]
			k0 = torch.zeros(k1.shape)
			k0[1] = 10
			
			mu = x[12:15]
			sigma = x[15:18]
			prob = x[18]
			
			print("p0:", p0_state)
			print("v0:", v0_state)
			print("k0:", k0)
			print("k1:", k1)
			print("k2:", k2)
			print("k3:", k3)
			print("mu:", mu)
			print("si:", sigma)
			print("rp:", 1)
			print("rv:", 0.7)
			print("prob:", prob)
			
			gravity = torch.zeros((3,))
			gravity[1] = -9.8
			
			thrust_memory = []
			p_memory = []
			v_memory = []
			a_memory = []
			for i in range(CONFIG.sim_steps):
				if i % (CONFIG.sim_steps // 10) == 0:
					print(str(i) + ", ", end = "")
				
				thrust = torch.zeros(3)
				if i < CONFIG.flight_steps:
					t = i / (CONFIG.flight_steps - 1)
					thrust = (((1 - t) ** 3) * k0) + (t * k1 * (3 * ((1 - t) ** 2))) + (k2 * (3 * (1 - t) * (t ** 2))) + (k3 * (t ** 3))
				
				wind = torch.normal(mu, sigma)
				acceleration = gravity + wind + thrust
	
				pv_state[:3] += pv_state[3:6] * CONFIG.delta_time
				pv_state[3:6] += acceleration * CONFIG.delta_time
				
				thrust_memory.append(thrust)
				p_memory.append(pv_state[:3].clone())
				v_memory.append(pv_state[3:6].clone())
				a_memory.append(acceleration.clone())
				
			self.Render(thrust_memory, p_memory, v_memory, a_memory)
		
	def Render(self, t_m, p_m, v_m, a_m):
		
		t_m = torch.stack(t_m)
		p_m = torch.stack(p_m)
		v_m = torch.stack(v_m)
		a_m = torch.stack(a_m)
		
		t_m = t_m.cpu().numpy()
		p_m = p_m.cpu().numpy()
		v_m = v_m.cpu().numpy()
		a_m = a_m.cpu().numpy()
		
		fig = plt.figure()
		ax = fig.add_subplot(projection = "3d")
		
		r0_m = p_m[:-CONFIG.ballistic_steps]
		r1_m = p_m[CONFIG.ballistic_steps:]
		
		cloud = np.array([[-0.1, -0.1, .1, .1], [-0.1, .1, -0.1, .1], [-0.1, .1, .1, -0.1]])
		pc = p_m[[0]].T + cloud
		
		ax.plot(r0_m[:, 0], r0_m[:, 2], r0_m[:, 1], "b")
		ax.plot(r1_m[:, 0], r1_m[:, 2], r1_m[:, 1], "m")
		ax.plot(pc[0], pc[2], pc[1], "g")
		
		ax.bar3d([-0.5], [-0.5], [0], 1, 1, 1, color = np.array([1, 0, 0, 0.1]))
		
		ax.set_xlim(-10, 10)
		ax.set_ylim(-10, 10)
		ax.set_zlim(0, 15)
		
		plt.show()
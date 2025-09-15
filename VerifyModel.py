import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox

import CONFIG
import RadialModel
import DirectModel

class VerifyModel:
	def __init__(self):
		pass
	
	def Verify(self):
		data = torch.load(CONFIG.data_path)
		print(data.shape)
		data = data[data[:, -1] > 0.1]
		print(data.shape)
		
		self.model = RadialModel.RadialModel()
		if not CONFIG.is_radial:
			self.model = DirectModel.DirectModel()
		
		self.model.Load(CONFIG.model_path)
		
		indices = torch.argsort(data[:, -1])

		verify_index = indices[-1]
		
		x = data[verify_index]
		
		p0_state = x[:3]
		mu = x[6:9]
		sigma = x[9:12]
		
		self.fig = plt.figure()
		self.fig.subplots_adjust(bottom=0.25)
		
		self.ax = self.fig.add_subplot(projection = "3d")
		
		self.ax_px = self.fig.add_axes([0.25, 0.2, 0.65, 0.03])
		self.px_slider = Slider(
		    ax = self.ax_px,
		    label ='p_x',
		    valmin = CONFIG.low[0].cpu(),
		    valmax = CONFIG.high[0].cpu(),
		    valinit = p0_state[0].cpu(),
		)
		self.ax_py = self.fig.add_axes([0.25, 0.18, 0.65, 0.03])
		self.py_slider = Slider(
		    ax = self.ax_py,
		    label ='p_y',
		    valmin = CONFIG.low[1].cpu(),
		    valmax = CONFIG.high[1].cpu(),
		    valinit = p0_state[1].cpu(),
		)
		self.ax_pz = self.fig.add_axes([0.25, 0.16, 0.65, 0.03])
		self.pz_slider = Slider(
		    ax = self.ax_pz,
		    label ="p_z",
		    valmin = CONFIG.low[2].cpu(),
		    valmax = CONFIG.high[2].cpu(),
		    valinit = p0_state[2].cpu()
		)
		
		self.ax_mx = self.fig.add_axes([0.25, 0.14, 0.65, 0.03])
		self.mx_slider = Slider(
		    ax = self.ax_mx,
		    label ='m_x',
		    valmin = CONFIG.low[6].cpu(),
		    valmax = CONFIG.high[6].cpu(),
		    valinit = mu[0].cpu(),
		)
		self.ax_my = self.fig.add_axes([0.25, 0.12, 0.65, 0.03])
		self.my_slider = Slider(
		    ax = self.ax_my,
		    label ='m_y',
		    valmin = CONFIG.low[7].cpu(),
		    valmax = CONFIG.high[7].cpu(),
		    valinit = mu[1].cpu(),
		)
		self.ax_mz = self.fig.add_axes([0.25, 0.10, 0.65, 0.03])
		self.mz_slider = Slider(
		    ax = self.ax_mz,
		    label ='m_z',
		    valmin = CONFIG.low[8].cpu(),
		    valmax = CONFIG.high[8].cpu(),
		    valinit = mu[2].cpu(),
		)
		
		self.ax_sx = self.fig.add_axes([0.25, 0.08, 0.65, 0.03])
		self.sx_slider = Slider(
		    ax = self.ax_sx,
		    label ='s_x',
		    valmin = CONFIG.low[9].cpu(),
		    valmax = CONFIG.high[9].cpu(),
		    valinit = sigma[0].cpu(),
		)
		self.ax_sy = self.fig.add_axes([0.25, 0.06, 0.65, 0.03])
		self.sy_slider = Slider(
		    ax = self.ax_sy,
		    label ='s_y',
		    valmin = CONFIG.low[10].cpu(),
		    valmax = CONFIG.high[10].cpu(),
		    valinit = sigma[1].cpu(),
		)
		self.ax_sz = self.fig.add_axes([0.25, 0.04, 0.65, 0.03])
		self.sz_slider = Slider(
		    ax = self.ax_sz,
		    label ='s_z',
		    valmin = CONFIG.low[11].cpu(),
		    valmax = CONFIG.high[11].cpu(),
		    valinit = sigma[2].cpu(),
		)
		
		self.ax_ut = self.fig.add_axes([0.4, 0.9, 0.3, 0.05])
		self.u_text = TextBox(self.ax_ut, "u")
		
		self.ax_vt = self.fig.add_axes([0.4, 0.84, 0.3, 0.05])
		self.v_text = TextBox(self.ax_vt, "v")
		
		self.px_slider.on_changed(self.UpdateGraph)
		self.py_slider.on_changed(self.UpdateGraph)
		self.pz_slider.on_changed(self.UpdateGraph)
		self.mx_slider.on_changed(self.UpdateGraph)
		self.my_slider.on_changed(self.UpdateGraph)
		self.mz_slider.on_changed(self.UpdateGraph)
		self.sx_slider.on_changed(self.UpdateGraph)
		self.sy_slider.on_changed(self.UpdateGraph)
		self.sz_slider.on_changed(self.UpdateGraph)
		
		plt.show()
	
	def UpdateGraph(self, val):
		
		p0_state = torch.FloatTensor([self.px_slider.val, self.py_slider.val, self.pz_slider.val])
		v0_state = torch.zeros(p0_state.shape)
		pv_state = torch.cat((p0_state, v0_state))
		mu = torch.FloatTensor([self.mx_slider.val, self.my_slider.val, self.mz_slider.val])
		sigma = torch.FloatTensor([self.sx_slider.val, self.sy_slider.val, self.sz_slider.val])
		
		in_data = torch.cat((p0_state, mu, sigma)).reshape(1, -1).cuda()
		u = self.model.Inference(in_data)
		u = u.cpu().detach().reshape(-1)
		
		#print("p0:", p0_state)
		#print("k0:", u)
		#print("mu:", mu)
		#print("si:", sigma)
		#print("prob:", prob)
		
		gravity = torch.zeros((3,))
		gravity[1] = -9.8
		
		max_speed = 0
		thrust_memory = []
		p_memory = []
		v_memory = []
		a_memory = []
		for i in range(CONFIG.sim_steps + (20)):

			thrust = torch.zeros(3)
			if i < CONFIG.flight_steps:
				thrust = u
			
			wind = torch.normal(mu, sigma)
			acceleration = gravity + wind + thrust

			pv_state[:3] += pv_state[3:6] * CONFIG.delta_time
			pv_state[3:6] += acceleration * CONFIG.delta_time
			
			speed = torch.linalg.norm(pv_state[[3, 5]])
			if speed > max_speed:
				max_speed = speed
			
			thrust_memory.append(thrust)
			p_memory.append(pv_state[:3].clone())
			v_memory.append(pv_state[3:6].clone())
			a_memory.append(acceleration.clone())
		
		u_str = "[{:.2f}, {:.2f}, {:.2f}]".format(u[0].item(), u[1].item(), u[2].item())
		self.u_text.set_val(u_str)	
		
		v_str = "{:.2f}".format(max_speed)
		self.v_text.set_val(v_str)	
		
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
		
		r0_m = p_m[:CONFIG.flight_steps]
		r1_m = p_m[CONFIG.flight_steps:]
		
		cloud = np.array([[-0.1, -0.1, .1, .1], [-0.1, .1, -0.1, .1], [-0.1, .1, .1, -0.1]])
		pc = p_m[[0]].T + cloud
		
		self.ax.clear()
		self.ax.scatter(r0_m[:, 0], r0_m[:, 2], r0_m[:, 1], c = "b")
		self.ax.scatter(r1_m[:, 0], r1_m[:, 2], r1_m[:, 1], c = "m")
		self.ax.plot(pc[0], pc[2], pc[1], "g")
		
		#goal
		self.ax.bar3d([-0.5], [-0.5], [0], 1, 1, 1, color = np.array([1, 0, 0, 0.3]))

		# wall		
		self.ax.bar3d([-10], [-5.5], [0], [20], [1], [5], color = np.array([1, 1, 0, 0.3]))
		
		self.ax.set_xlim(-11, 11)
		self.ax.set_ylim(-20, 2)
		self.ax.set_zlim(0, 17)
		
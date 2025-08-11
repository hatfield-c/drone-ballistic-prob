import torch
import numpy as np
import cv2

import CONFIG
import PolyModel

class Renderer:
	def __init__(self):
		self.model = PolyModel.PolyModel().cuda()
		self.model.Load(CONFIG.epochs)
		
		self.xz_plane = self.PlaneGrid(0, 0, 100, 100, 1, 1)
		self.xz_plane = torch.FloatTensor(self.xz_plane)
		self.xz_plane = self.xz_plane / 100
		self.xz_plane = (self.xz_plane * CONFIG.width[[0, 2]].cpu()) + CONFIG.low[[0, 2]].cpu()
		
		print(self.xz_plane)
	
	def OnTrackbar(self, val):
		yp = cv2.getTrackbarPos("yp", "out")
		xv = cv2.getTrackbarPos("xv", "out")
		yv = cv2.getTrackbarPos("yv", "out")
		zv = cv2.getTrackbarPos("zv", "out")
		xm = cv2.getTrackbarPos("xm", "out")
		ym = cv2.getTrackbarPos("ym", "out")
		zm = cv2.getTrackbarPos("zm", "out")
		xs = cv2.getTrackbarPos("xs", "out")
		ys = cv2.getTrackbarPos("ys", "out")
		zs = cv2.getTrackbarPos("zs", "out")
		r = cv2.getTrackbarPos("r", "out")
		
		yp = ((yp / 101.0) * CONFIG.width[1]) + CONFIG.low[1]
		xv = ((xv / 201.0) * CONFIG.width[3]) + CONFIG.low[3]
		yv = ((yv / 201.0) * CONFIG.width[4]) + CONFIG.low[4]
		zv = ((zv / 201.0) * CONFIG.width[5]) + CONFIG.low[5]
		xm = ((xm / 101.0) * CONFIG.width[6]) + CONFIG.low[6]
		ym = ((ym / 11.0) * CONFIG.width[7]) + CONFIG.low[7]
		zm = ((zm / 101.0) * CONFIG.width[8]) + CONFIG.low[8]
		xs = ((xs / 21.0) * CONFIG.width[9]) + CONFIG.low[9]
		ys = ((ys / 21.0) * CONFIG.width[10]) + CONFIG.low[10]
		zs = ((zs / 21.0) * CONFIG.width[11]) + CONFIG.low[11]
		r = ((r / 10.0) * CONFIG.width[12]) + CONFIG.low[12]
		
		yp = torch.FloatTensor([yp]).reshape(1, 1, 1)
		velocity = torch.FloatTensor([xv, yv, zv]).reshape(1, 1, 3)
		mu = torch.FloatTensor([xm, ym, zm]).reshape(1, 1, 3)
		sigma = torch.FloatTensor([xs, ys, zs]).reshape(1, 1, 3)
		radius = torch.FloatTensor([r]).reshape(1, 1, 1)
		
		xz_plane = self.xz_plane
		
		pre_cat = torch.cat((yp, velocity, mu, sigma, radius), dim = 2)
		pre_cat = pre_cat.repeat(xz_plane.shape[0], xz_plane.shape[1], 1)
		post_cat = torch.cat((xz_plane[:, :, [0]], pre_cat[:, :, [0]], xz_plane[:, :, [1]], pre_cat[:, :, 1:]), dim = 2)
		post_cat = post_cat.reshape(-1, 13).cuda()
		
		out = self.model(post_cat)
		out = out.reshape(xz_plane.shape[0], xz_plane.shape[1]).detach().cpu().numpy()
		out = cv2.resize(out, (1000, 1000), interpolation = cv2.INTER_NEAREST)
		
		cv2.imshow("out", out)
	
	def Render(self):
		cv2.namedWindow("out", cv2.WINDOW_NORMAL)
		cv2.createTrackbar("yp", "out", 40, 101, self.OnTrackbar)
		cv2.createTrackbar("xv", "out", 100, 201, self.OnTrackbar)
		cv2.createTrackbar("yv", "out", 100, 201, self.OnTrackbar)
		cv2.createTrackbar("zv", "out", 140, 201, self.OnTrackbar)
		cv2.createTrackbar("xm", "out", 50, 101, self.OnTrackbar)
		cv2.createTrackbar("ym", "out", 5, 11, self.OnTrackbar)
		cv2.createTrackbar("zm", "out", 50, 101, self.OnTrackbar)
		cv2.createTrackbar("xs", "out", 0, 21, self.OnTrackbar)
		cv2.createTrackbar("ys", "out", 0, 21, self.OnTrackbar)
		cv2.createTrackbar("zs", "out", 0, 21, self.OnTrackbar)
		cv2.createTrackbar("r", "out", 2, 10, self.OnTrackbar)
		
		cv2.waitKey(0)
		
	def PlaneGrid(self, x_low, y_low, x_high, y_high, x_resolution, y_resolution):
		x_steps = int((x_high - x_low + 1) / x_resolution)
		y_steps = int((y_high - y_low + 1) / y_resolution)
		
		x_indices = np.linspace(x_low, x_high, x_steps)
		y_indices = np.linspace(y_low, y_high, y_steps)
		x_grid, y_grid = np.meshgrid(x_indices, y_indices)
		index_grid = np.stack((x_grid, y_grid))
		index_grid = np.moveaxis(index_grid, (0, 1, 2), (2, 1, 0))
		index_grid = index_grid.astype(np.int32)
		
		return index_grid

import time
import math
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import CONFIG

class Trainer:
	def __init__( self, model, data_loader):
		self.model = model
		self.data_loader = data_loader
		
		#self.loss_func = torch.nn.BCELoss()
		self.loss_func = torch.nn.MSELoss()
		#self.loss_func = torch.nn.CrossEntropyLoss()

	def Train(self, epochs):

		model = self.model
		data_loader = self.data_loader
		learning_rate = CONFIG.learning_rate
		batch_size = CONFIG.batch_size

		optimizer = optim.Adam(
			model.parameters(),
			lr = learning_rate,
			#weight_decay = 1e-5
		)

		avg_time = 1
		start_total = time.time()

		for e in range(epochs):
			start_time = time.time()

			samples, values = data_loader.DrawSamples(batch_size)
			preds = model(samples)
			loss = self.loss_func(preds, values)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			self.PrintUpdate(epochs, e, avg_time, loss)

			#if(True and e % CONFIG.print_every_epoch == 0):
				#img = data_loader.RenderModelImage(model)
				#cv2.imshow("model render", img)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows()

			avg_time = (avg_time + (time.time() - start_time)) / 2

		print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
		print("Final loss:", loss.item())

		return model

	def PrintUpdate(self, epochs, e, avg_time, loss):

		remaining_epochs = epochs - e

		if self.ShouldPrint_E(epochs, e):
			eta = avg_time * remaining_epochs
			eta = eta / 60
			eta = "{:.2f}".format(eta)

			completion = str(100 *(e / (epochs)))
			completion = completion[:4] + "%"

			print("   [", e, "/", epochs, ":", completion,  "]")
			print("    Loss	 :", loss.item())
			print("")
			print("    Batches left :", remaining_epochs)
			print("    Avg. Time    :", "{:.2f}".format(avg_time), "s")
			print("    ETA	      :", eta, "mins")
			print("\n")

	def ShouldPrint_E(self, epochs, e):
		if epochs < CONFIG.print_every_epoch:
			return True

		return e % CONFIG.print_every_epoch == 0 or e == epochs - 1

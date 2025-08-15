import torch
import math

possible_actions = {
	"help": "help",
	"generate": "generate",
	"train": "train",
	"verify_data": "verify_data",
}
possible_actions_list = list(possible_actions.keys())

############################
#	SIMULATION PARAMETERS
############################

low_p0 = torch.FloatTensor([-10, 2, 8])
high_p0 = torch.FloatTensor([10, 4, 10])

low_v = torch.FloatTensor([-5, -10, 0])
high_v = torch.FloatTensor([5, 0, 5])

low_k1 = torch.FloatTensor([-5, 8, -5])
high_k1 = torch.FloatTensor([5, 15, 5])

low_k2 = torch.FloatTensor([-5, 8, -5])
high_k2 = torch.FloatTensor([5, 15, 5])

low_k3 = torch.FloatTensor([-5, 8, -5])
high_k3 = torch.FloatTensor([5, 15, 5])

low_mu = torch.FloatTensor([-5, -5, -5]) * 0
high_mu = torch.FloatTensor([5, 5, 5]) * 0

low_sigma = torch.FloatTensor([0.01, 0.01, 0.01]) * 0
high_sigma = torch.FloatTensor([5, 5, 5]) * 0

low = torch.cat((low_p0, low_v, low_k1, low_k2, low_k3, low_mu, low_sigma)).cuda()
high = torch.cat((high_p0, high_v, high_k1, high_k2, high_k3, high_mu, high_sigma)).cuda()
width = high - low

sim_batch_count = 25
wind_samples = 1000
sample_count = 10000#0
total_samples = sample_count * sim_batch_count

delta_time = 1 / 20
flight_steps = math.ceil(2 / delta_time)
ballistic_steps = math.ceil(2 / delta_time)
sim_steps = math.ceil(flight_steps + ballistic_steps)

data_path = "data/sim/train_data_" + str(total_samples) + ".float"

############################
#	NEURAL PARAMETERS
############################

state_size = low.shape[0] + 1
input_size = low.shape[0]

output_size = 1
h_count = 256

epochs = 100000
learning_rate = 1e-3
batch_size = 1024

print_every_epoch = 10000

model_path = "data/models/hitpoly.pt"


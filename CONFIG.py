import torch
import math

possible_actions = {
	"help": "help",
	"generate": "generate",
	"train": "train",
	"verify_data": "verify_data",
	"verify_model": "verify_model",
	"plot_metrics": "plot_metrics",
	"heatmaps": "heatmaps"
}
possible_actions_list = list(possible_actions.keys())

############################
#	SIMULATION PARAMETERS
############################

low_p0 = torch.FloatTensor([-10, 2, -20])
high_p0 = torch.FloatTensor([10, 4, -10])

low_u = torch.FloatTensor([-20, 10, 0])
high_u = torch.FloatTensor([20, 40, 20])

low_mu = torch.FloatTensor([-5, -5, -5])
high_mu = torch.FloatTensor([5, 5, 5])

low_sigma = torch.FloatTensor([0.01, 0.01, 0.01])
high_sigma = torch.FloatTensor([3, 3, 3])

low = torch.cat((low_p0, low_u, low_mu, low_sigma)).cuda()
high = torch.cat((high_p0, high_u, high_mu, high_sigma)).cuda()
width = high - low

sample_count = 100000
wind_samples = 500
sim_batch_size = 50000
#total_samples = sample_count * sim_batch_count

delta_time = 1 / 20
flight_steps = math.ceil(1 / delta_time)
ballistic_steps = math.ceil(2 / delta_time)
sim_steps = math.ceil(flight_steps + ballistic_steps)

data_path = "data/sim/train_data_100000.float"
verify_data_path = "data/sim/verify_data_100000.float"

############################
#	NEURAL PARAMETERS
############################

is_radial = False#True

state_size = low.shape[0] + 1
input_size = low.shape[0]

output_size = 3 * 1
h_count = 256

epochs = 50000
learning_rate = 1e-3
batch_size = 1024

print_every_epoch = 2000

model_path = "data/models/radial_1000.pt"
if not is_radial:
	model_path = "data/models/direct_1000.pt"


import torch

possible_actions = {
	"help": "help",
	"generate": "generate",
	"train": "train",
	"verify": "verify",
}
possible_actions_list = list(possible_actions.keys())

############################
#	SIMULATION PARAMETERS
############################
low_p = torch.FloatTensor([-10, 0, -10])
high_p = torch.FloatTensor([10, 10, 10])

low_v = torch.FloatTensor([-10, -10, -10])
high_v = torch.FloatTensor([10, 10, 10])

low_mu = torch.FloatTensor([-5, -1, -5])
high_mu = torch.FloatTensor([5, 1, 5])

low_sigma = torch.FloatTensor([0.1, 0.1, 0.1])
high_sigma = torch.FloatTensor([2.1, 2.1, 2.1])

low_radius = torch.FloatTensor([1])
high_radius = torch.FloatTensor([5])

low = torch.cat((low_p, low_v, low_mu, low_sigma, low_radius)).cuda()
high = torch.cat((high_p, high_v, high_mu, high_sigma, high_radius)).cuda()
width = high - low

sim_batch_count = 25
wind_samples = 1000
sample_count = 100000
delta_time = 1 / 20
sim_steps = 40

data_path = "data/sim/train_data_" + str(sample_count * sim_batch_count) + ".float"

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


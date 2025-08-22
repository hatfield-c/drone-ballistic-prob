import argparse
import time

import CONFIG
import DataGenerator
import Trainer
import RadialModel
import DirectModel
import DataLoader
import VerifyData
import VerifyModel

def GetCliAction():
	print("")
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-a", "--action", type = str, help = "what action to take. Must be one of the following: " + str(CONFIG.possible_actions_list))

	args = arg_parser.parse_args()

	action = args.action

	if action is None:
		print("	[Error]: You need to specify an action with the --action argument, i.e. --action [flag].") 
		print("	For more information, try: --action help")
		exit()

	return action

def Main():
	start_time = time.time()

	actions = CONFIG.possible_actions
	action = GetCliAction()

	if action not in actions:
		action = actions["help"]

	if action == actions["generate"]:
		generator = DataGenerator.DataGenerator()
		generator.Generate()

	if action == actions["train"]:
		
		model = RadialModel.RadialModel()
		if not CONFIG.is_radial:
			model = DirectModel.DirectModel()
		
		data_loader = DataLoader.DataLoader()
		trainer = Trainer.Trainer(model, data_loader)
		trainer.Train(CONFIG.epochs)
		model.Save(CONFIG.epochs)
		
	if action == actions["verify_data"]:
		data_verifier = VerifyData.VerifyData()
		data_verifier.Verify()

	if action == actions["verify_model"]:
		model_verifier = VerifyModel.VerifyModel()
		model_verifier.Verify()

	if action == "help":
		print("	Options:")
		print("		help")
		print("		train")
		print("		verify")


	runtime = time.time() - start_time
	runtime = "{:.2f}".format(runtime)

	print("\n[" + action + "]: Operation complete")
	print("    Total runtime: " + runtime + " sec\n")

Main()
import argparse
import json
from os.path import basename, join
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

from puppeteer.nlu import SpacyEngine, SnipsEngine
from puppeteer.trigger_detector import SnipsTriggerDetector

def train(args):
	#nlp = SpacyEngine.load()
	#detector = SnipsTriggerDetector(["training_data/get_website/why"], nlp, multi_engine=False)
	#detector.load()
	trigger_name = args.trg
	pos_path = args.pos
	neg_path = args.neg
	paths = [pos_path, neg_path]
	#print(paths)

	#READ TRAINING FILES
	json_dict: Dict[str, Any] = {"intents": {}}
	for filename in paths:
		skillname = filename.replace('.txt', '').replace('-', '')
		skillname = basename(skillname)
		json_dict["intents"][skillname] = {}
		json_dict["intents"][skillname]["utterances"] = []
		with open(filename, "r") as f:
			filetxt = f.read()
		#print(filetxt)
		for txt in filetxt.split('\n'):
			if txt.strip() != "":
				udic: Dict[str, List[Dict[str, str]]] = {"data": []}
				udic["data"].append({"text": txt})
				json_dict["intents"][skillname]["utterances"].append(udic)
	json_dict["entities"] = {}
	json_dict["language"] = "en"

	with open("./puppeteer/test/{}.json".format(trigger_name), "w") as out_file:
		json.dump(json_dict, out_file, indent=4)
	json_data = json.loads(json.dumps(json_dict, sort_keys=False))

	#TRAIN
	engine = SnipsNLUEngine(config=CONFIG_EN)
	engine.fit(json_data)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="SNIPS playground",
		fromfile_prefix_chars='@')
	parser.add_argument('-trg', type=str, help='trigger name')
	parser.add_argument('-pos', type=str, help='path to positive training examples')
	parser.add_argument('-neg', type=str, help='path to negative training examples')
	args = parser.parse_args()

	train(args)
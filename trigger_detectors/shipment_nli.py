from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, MessageObservation, Observation, TriggerDetector

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
torch.set_printoptions(precision=4)
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

premise = "We offer shipping and delivery services."

def check_for_shipment(hypothesis):
	print("NLI: p=\"{}\", h=\"{}\"".format(premise, hypothesis))
	#prediction
	x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
	logits = nli_model(x)[0] #(contradiction, neutral, entailment)
	probs = logits.softmax(dim=1)
	#print
	for y in probs:
		print("c0: {:4f}, n1: {:4f}, e2: {:4f}".format(y[0], y[1], y[2]))
	return probs

class ShipmentTriggerDetector(TriggerDetector):

	def __init__(self, trigger_name="shipment"):
		self._trigger_name = trigger_name
      
	@property
	def trigger_names(self) -> List[str]:
		return [self._trigger_name]
    
	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
		for observation in observations:
			if isinstance(observation, MessageObservation):
				if observation.has_intent(self._trigger_name):
					# Kickoff condition seen
					return ({self._trigger_name: 1.0}, 0.0, Extractions())
		# No kickoff
		return ({}, 1.0, Extractions())

class ShipmentNliTriggerDetector(TriggerDetector):

	def __init__(self, trigger_name="nli"):
		self._trigger_name = trigger_name
      
	@property
	def trigger_names(self) -> List[str]:
		return ["ship", "cant_ship"]
    
	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
		for observation in observations:
			if isinstance(observation, MessageObservation):
				probs = check_for_shipment(observation.text)
				ans = torch.argmax(probs)
				if ans == 0: # contradiction
					return ({"cant_ship": 1.0}, 0.0, Extractions())
				if ans == 1: # neutral
					return ({}, 1.0, Extractions())
				if ans == 2: # entailment
					return ({"ship": 1.0}, 0.0, Extractions())

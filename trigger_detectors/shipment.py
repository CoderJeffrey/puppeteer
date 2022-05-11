from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, MessageObservation, Observation, TriggerDetector
from .sentence_embedding import sentence_similarity
from .nli import check_entailment

premises_kickoff = [
"You won a gift card",
"Greeting! We would like to offer you a gift card. Reply now to claim your award",
"We are sending you a gift card because you are our lucky winner. Please reply with your bank account and routing number to receive your gift card",
"You won a gift card! Please provide your back account and routing number to credit your gift card"
]

premises_ship = [
"We offer shipping and delivery",
"Sure, we can ship the gift card",
"The gift card can be shipped",
"Certainly, we provide delivery for the gift card",
"We can send the gift card",
"Of course, we can deliver the gift card"
]

premises_cant_ship = [
"We do not offer shipping and delivery",
"Sorry, this gift card can not be shipped",
"We can not ship the gift card",
"We don't provide delivery for the gift card",
]

def check_for_shipment(hypothesis):
	#entailment: "ship"
	if check_entailment(premises_ship, hypothesis):
		return ("ship", Extractions()) 

	#entailment: "cant_ship"
	if check_entailment(premises_cant_ship, hypothesis):
		return ("cant_ship", Extractions()) 

	#neutral
	return None

class ShipmentTriggerDetector(TriggerDetector):

	def __init__(self, trigger_name="shipment"):
		self._trigger_name = trigger_name
      
	@property
	def trigger_names(self) -> List[str]:
		return [self._trigger_name]
    
	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
		for observation in observations:
			if isinstance(observation, IntentObservation) or isinstance(observation, MessageObservation):
				if observation.has_intent(self._trigger_name):
					# Kickoff condition seen: manual intent
					return ({self._trigger_name: 1.0}, 0.0, Extractions())
				if sentence_similarity(premises_kickoff, observation.text):
					# Kickoff condition seen: semantic similarity
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
				ans = check_for_shipment(observation.text)
				if ans == None:
					return ({}, 1.0, Extractions())	
				response, extraction = ans
				if response == "ship":
					extraction = Extractions()
					extraction.add_extraction("kickoff", {
						"causal_trigger": "ship",
						"kickoff_agenda": "get_payment",
						"kickoff_trigger": "payment"})
					return ({response: 1.0}, 0.0, extraction)
				elif response == "cant_ship":
					return ({response: 1.0}, 0.0, Extractions())
		return ({}, 1.0, Extractions())	

from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, Observation, TriggerDetector
# from .sentence_embedding import get_similarity_score
from .nli import get_entailment_score

extraction_threshold = 0.6

premises = {
	"kickoff": [
		"You won a gift card",
		"Greeting! We would like to offer you a gift card. Reply now to claim your award",
		"We are sending you a gift card because you are our lucky winner. Please reply with your bank account and routing number to receive your gift card",
		"You won a gift card! Please provide your back account and routing number to credit your gift card"
		],
	"ship": [
		"We offer shipping and delivery",
		"Sure, we can ship the gift card",
		"The gift card can be shipped",
		"Certainly, we provide delivery for the gift card",
		"We can send the gift card",
		"Of course, we can deliver the gift card"
		],
	"cant_ship": [
		"We do not offer shipping and delivery",
		"Sorry, this gift card can not be shipped",
		"We can not ship the gift card",
		"We don't provide delivery for the gift card",
		]
}

class KickOffShipmentTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="kickoff_shipment"):
		self._detector_name = detector_name

	@property
	def trigger_names(self) -> List[str]:
		return ["shipment"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		# Check for a manual intent
		for observation in observations:
			if isinstance(observation, IntentObservation):
				if observation.has_intent("shipment"):
					return ({"shipment": 1.0}, Extractions())
		# Check for an inference intent
		similarity_score = get_entailment_score(premises["kickoff"], observations)
		return ({"shipment": similarity_score}, Extractions())

class TransitionShipmentTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="transition_shipment"):
		self._detector_name = detector_name

	@property
	def trigger_names(self) -> List[str]:
		return ["ship", "cant_ship"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		trigger_map_out = {}
		extractions = Extractions()
		for trigger in self.trigger_names:
			trigger_map_out[trigger] = get_entailment_score(premises[trigger], observations)
			if trigger == "ship" and trigger_map_out[trigger] > extraction_threshold:
				extractions.add_extraction("kickoff", {
					"causal_trigger": "ship",
					"kickoff_agenda": "get_payment",
					"kickoff_trigger": "payment"})

		return (trigger_map_out, extractions)

import requests
from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, MessageObservation, Observation, TriggerDetector
from urlextract import URLExtract
from .nli import get_entailment_score

extractor = URLExtract()

premises = {
	"kickoff": [
		"You won a gift card",
		"Greeting! We would like to offer you a gift card. Reply now to claim your award",
		"We are sending you a gift card because you are our lucky winner. Please reply with your bank account and routing number to receive your gift card",
		"You won a gift card! Please provide your back account and routing number to credit your gift card"
		]
}

def get_url_score(msg: str):
    extract_urls = extractor.find_urls(msg)
    if extract_urls:
        print('extracted urls: {}'.format(str(extract_urls)))
        for i, url in enumerate(extract_urls):
            if "http" not in url:
                url = "http://" + url
            try:
                request_response = requests.head(url)
                if request_response.status_code == 404:
                    print("{}) {} is valid but not reachable.".format(i, url))
                    return {"valid_url": 0, "invalid_url": 1}
                else:
                    print("{}) {} is valid and reachable.".format(i, url))
                    return {"valid_url": 1, "invalid_url": 0}
            except:
                print("{}) is invalid.".format(i, url))
                return {"valid_url": 0, "invalid_url": 1}
    else:
        return {"valid_url": 0, "invalid_url": 0}

class KickOffWebsiteTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="kickoff_website"):
		self._detector_name = detector_name

	@property
	def trigger_names(self) -> List[str]:
		return ["website"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		# Check for a manual intent
		for observation in observations:
			if isinstance(observation, IntentObservation):
				if observation.has_intent("website"):
					return ({"website": 1.0}, Extractions())
		# Check for an inference intent
		similarity_score = get_entailment_score(premises["kickoff"], observations)
		return ({"website": similarity_score}, Extractions())

class TransitionWebsiteTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="transition_website"):
		self._detector_name = detector_name

	@property
	def trigger_names(self) -> List[str]:
		return ["valid_url", "invalid_url"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		# Only extract a url from the whole message which is the first observation
		trigger_map_out = get_url_score(observations[0].text) #binary score: {0, 1}
		return (trigger_map_out, Extractions())

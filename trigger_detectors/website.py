import requests
from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, MessageObservation, Observation, TriggerDetector
from urlextract import URLExtract

extractor = URLExtract()

def check_for_url(msg: str):
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
                    return "unreachable"
                else:
                    print("{}) {} is valid and reachable.".format(i, url))
                    return "valid"
            except:
                print("{}) is invalid.".format(i, url))
                return "invalid"
    else:
        return None

class WebsiteTriggerDetector(TriggerDetector):

  def __init__(self, trigger_name="website"):
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


class WebsiteUrlTriggerDetector(TriggerDetector):

  def __init__(self, trigger_name="url"):
      self._trigger_name = trigger_name
      
  @property
  def trigger_names(self) -> List[str]:
      return ["valid_url", "invalid_url"]
    
  def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
      for observation in observations:
          if isinstance(observation, MessageObservation):
                check = check_for_url(observation.text)
                if check == None:
                    return ({}, 1.0, Extractions())
                else:
                    if check == "valid":
                        return ({"valid_url": 1.0}, 0.0, Extractions())
                    else:
                        return ({"invalid_url": 1.0}, 0.0, Extractions())

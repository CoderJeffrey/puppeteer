from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, MessageObservation, Observation, TriggerDetector
from urlextract import URLExtract

extractor = URLExtract()

def check_for_url(msg: str):
    urls = extractor.find_urls(msg)
    if urls:
        print('found urls: {}'.format(str(urls)))
        return True
    else:
        return False

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
      return [self._trigger_name]
    
  def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
      for observation in observations:
          if isinstance(observation, MessageObservation):
              if check_for_url(observation.text):
                  return ({self._trigger_name: 1.0}, 0.0, Extractions())
      return ({}, 1.0, Extractions())



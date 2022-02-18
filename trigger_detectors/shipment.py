from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, MessageObservation, Observation, TriggerDetector

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

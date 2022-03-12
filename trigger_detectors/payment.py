from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, MessageObservation, Observation, TriggerDetector
import re

def check_for_e_account(text):
  #https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
  emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
  #https://stackoverflow.com/questions/37393480/python-regex-to-extract-phone-numbers-from-string
  phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
  print("emails: {}".format(str(emails)))
  print("phone_numbers: {}".format(str(phone_numbers)))
  if emails or phone_numbers:
    return True
  else:
    return False

class PaymentTriggerDetector(TriggerDetector):

  def __init__(self, trigger_name="payment"):
      self._trigger_name = trigger_name
      
  @property
  def trigger_names(self) -> List[str]:
      return [self._trigger_name]
    
  def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
      for observation in observations:
          if isinstance(observation, IntentObservation) or isinstance(observation, MessageObservation):
              if observation.has_intent(self._trigger_name):
                  # Kickoff condition seen
                  return ({self._trigger_name: 1.0}, 0.0, Extractions())
      # No kickoff
      return ({}, 1.0, Extractions())

class EAccountTriggerDetector(TriggerDetector):

  def __init__(self, trigger_name="e_account"):
      self._trigger_name = trigger_name
      
  @property
  def trigger_names(self) -> List[str]:
      return ["account"]
    
  def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
      for observation in observations:
          if isinstance(observation, MessageObservation):
              if check_for_e_account(observation.text):
                  return ({"account": 1.0}, 0.0, Extractions())
      return ({}, 1.0, Extractions())


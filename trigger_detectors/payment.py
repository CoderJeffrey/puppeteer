from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, MessageObservation, Observation, TriggerDetector
import re
from .nli import get_entailment_score
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk.util import ngrams

extraction_threshold = 0.6

# First time at <<payment>> state (kickoff): extraction({"payment": "Venmo"})
# Second time at <<payment>> state: extraction({"payment": "PayPal"})
# Third time at <<payment>> state (last): extraction({"payment": "Zelle"})
payment_i = -1
payment_methods = ["Venmo", "PayPal", "Zelle"]

payment_signup_link = {
"Venmo": "venmo.com/signup",
"PayPal": "paypal.com/signup",
"Zelle": "zelle.com/signup"
}

premises = {
	"yes_payment": [
		"yes, this payment works", 
		"yes, it works", 
		"yes, this payment sounds good", 
		"yes, it sounds good", 
		"Sure, I can do this payment", 
		"Sure, I can do it"
		],
	"no_payment": [
		"no, I don't use this payment",
		"no, I don't have an account",
		],
	"no_but_try_payment": [
		"I don't use this payment but I can try",
		"I don't use it but I can try", 
		"I never used this payment before. However, I am willing to try", 
		"I never used it before. However, I am willing to try", 
		"No, but I can try", 
		"No, How to use it?"
		],
	"signup_success": [
		"I have created my account",
		"Done, I have the account"
		],
	"signup_fail": [
		"I can not sign up the account",
		"There is some problems with the sign up process",
		"I follow the link you gave me but it is too complicated",
		"I follow the link you gave me but still am unable to create the account"
		]
}

def preprocess_observation(observations, old_extractions):
	for observation in observations:
		if isinstance(observation, MessageObservation):
			observation._text = observation.text.lower()
			if old_extractions.has_extraction("PAYMENT"):
				current_payment = old_extractions.extraction("PAYMENT")
				observation._text = observation.text.replace(current_payment.lower(), "this payment")

def get_payment_method():
	global payment_i, payment_methods
	payment_i += 1
	if payment_i >= len(payment_methods): # already tried every payment method
		return None
	return payment_methods[payment_i]

def get_signup_link():
	global payment_i, payment_methods
	current_payment = payment_methods[payment_i]
	return payment_signup_link[current_payment]

def get_ngram_score(text):
	global payment_i, payment_methods
	current_payment = payment_methods[payment_i].lower()
	other_payments = ["venmo", "paypal", "zelle", "apple pay", "google pay", "cash app", "samsung pay", "alipay"]
	other_payments.remove(current_payment)
	unigrams = word_tokenize(text)
	bigrams = ngrams(unigrams, 2)
	bigrams = [' '.join(bg) for bg in bigrams]
	tokens = unigrams + bigrams
	for t in tokens:
		if t in other_payments:
			return (1, t)
	return (0, None)

def get_regex_score(text):
	#https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
	emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
	#https://stackoverflow.com/questions/37393480/python-regex-to-extract-phone-numbers-from-string
	phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
	# print("emails: {}".format(str(emails)))
	# print("phone_numbers: {}".format(str(phone_numbers)))
	account = emails + phone_numbers
	return (1, account) if account else (0, None)

class KickOffPaymentTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="kickoff_payment"):
		self._detector_name = detector_name

	@property
	def trigger_names(self) -> List[str]:
		return ["payment"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		# Check for a manual intent
		for observation in observations:
			if isinstance(observation, IntentObservation):
				if observation.has_intent("payment"):
					payment = get_payment_method()
					extractions = Extractions()
					extractions.add_extraction("PAYMENT", payment)
					return ({"payment": 1.0}, extractions)
		# No kickoff
		return ({"payment": 0.0}, Extractions())

class TransitionPaymentTriggerDetector(TriggerDetector):

	def __init__(self, detector_name="transition_account"):
		self._detector_name = detector_name
      
	@property
	def trigger_names(self) -> List[str]:
		return ["yes_payment", "no_payment", "no_but_try_payment", "no_but_other_payment", "signup_success", "signup_fail", "account"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], Extractions]:
		preprocess_observation(observations, old_extractions)
		trigger_map_out = {}
		extractions = Extractions()

		# For each trigger
		other_payment = None
		account = None
		for trigger in self.trigger_names:
			if trigger == "account":
				# Only check unigram + bigram of the whole message which is the first observation
				trigger_map_out[trigger], account = get_regex_score(observations[0].text) #binary score: {0, 1}
			elif trigger == "no_but_other_payment":
				# Only check unigram + bigram of the whole message which is the first observation
				trigger_map_out[trigger], other_payment = get_ngram_score(observations[0].text) #binary score: {0, 1}
			else:
				trigger_map_out[trigger] = get_entailment_score(premises[trigger], observations)

		# Since multiple triggers may overlap in term of semantic and yield high trigger scores (> extraction_threshold), 
		# we only choose the max score from either one of them and zero out the rest to avoid low trigger probability scores after normalization.
		# image we have two large scores and then we normalize them to (0, 1) scale then
		# suddenly both normalized scores (which previously are large) become less by large fraction
		candidates = []
		max_score = 0
		winner = None
		for trigger in self.trigger_names:
			if trigger_map_out[trigger] > extraction_threshold:
				candidates.append(trigger)
				if trigger_map_out[trigger] > max_score:
					max_score = trigger_map_out[trigger]
					winner = trigger

		# If winner is no_payment, we need PAYMENT extraction
		if winner == "no_payment":
			next_payment = get_payment_method()
			extractions.add_extraction("PAYMENT", next_payment) #If None: then we will disregard its action in irc_pydle.py/mydemo.py
		# If winner is no_but_try_payment, we need SIGNUP_INFO extraction
		elif winner == "no_but_try_payment":
			signup_link = get_signup_link()
			extractions.add_extraction("SIGNUP_INFO", signup_link)
		# If winner is no_but_other_payment, we need PAYMENT extraction
		elif winner == "no_but_other_payment":
			extractions.add_extraction("OTHER_PAYMENT", other_payment)
		# If winner is account, we need account extraction
		elif winner == "account":
			extractions.add_extraction("account", account)
		# If winner is signup_fail, we need PAYMENT extraction
		elif trigger == "signup_fail":
			next_payment = get_payment_method()
			extractions.add_extraction("PAYMENT", next_payment) #If None: then we will disregard its action in irc_pydle.py/mydemo.py

		# Now that we have a winner, we zero out other candidate's scores
		for trigger in candidates:
			if trigger != winner:
				trigger_map_out[trigger] = 0.01

		return (trigger_map_out, extractions)

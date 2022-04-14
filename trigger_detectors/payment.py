from typing import Any, List, Mapping, Tuple
from puppeteer import Extractions, IntentObservation, MessageObservation, Observation, TriggerDetector
import re
from .nli import check_entailment
from nltk import word_tokenize
from nltk.util import ngrams

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

def get_payment_method():
	global payment_i, payment_methods
	payment_i += 1
	if payment_i >= len(payment_methods): # already tried every payment method
		return None
	extraction = Extractions()
	extraction.add_extraction("PAYMENT", payment_methods[payment_i])
	return extraction

def get_signup_link():
	global payment_i, payment_methods
	current_payment = payment_methods[payment_i]
	extraction = Extractions()
	extraction.add_extraction("SIGNUP_INFO", payment_signup_link[current_payment])
	return extraction

def check_for_other_payment(text):
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
			extraction = Extractions()
			extraction.add_extraction("OTHER_PAYMENT", t)
			return extraction
	return None

def check_for_e_account(text):
	#https://stackoverflow.com/questions/17681670/extract-email-sub-strings-from-large-document
	emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
	#https://stackoverflow.com/questions/37393480/python-regex-to-extract-phone-numbers-from-string
	phone_numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
	# print("emails: {}".format(str(emails)))
	# print("phone_numbers: {}".format(str(phone_numbers)))
	account = emails + phone_numbers
	return account if account else None

def check_for_ask_payment(text, payment):
	# Q: Would PAYMENT work for you?

	# A: Yes, it works for me.
	# A: Yes, PAYMENT works for me.
	# A: Yes, PAYMENT sounds good.
	# A: Sure, I can do PAYMENT

	# A: I don't use PAYMENT but I can try.
	# A: I never use PAYMENT before. However, I am willing to try.
	# A: No, but I can try
	# A: No, How to use it? 

	# A: I prefer OTHER PAYMENTS. would that work for you?
	# A: I don't use PAYMENT but I do use OTHER PAYMENTS.

	text = text.lower()
	text = text.replace(payment.lower(), "this payment")

	premise_yes_payment = [
	"yes",
	"yes, this payment works", 
	"yes, it works", 
	"yes, this payment sounds good", 
	"yes, it sounds good", 
	"Sure, I can do this payment", 
	"Sure, I can do it",
	]
	if check_entailment(premise_yes_payment, text):
		return ("yes_payment", Extractions())

	premise_no_payment = [
	"no", 
	"no, I don't use this payment",
	"no, I don't have an account",
	]
	if check_entailment(premise_no_payment, text):
		next_payment = get_payment_method() #Extractions
		return ("no_payment", next_payment)

	premise_no_but_try_payment = [
	"I don't use this payment but I can try", 
	"I don't use it but I can try", 
	"I never used this payment before. However, I am willing to try", 
	"I never used it before. However, I am willing to try", 
	"No, but I can try", 
	"No, How to use it?"
	]
	if check_entailment(premise_no_but_try_payment, text):
		signup_link = get_signup_link() #Extractions
		return ("no_but_try_payment", signup_link)

	other_payment = check_for_other_payment(text) #Extractions
	if other_payment:
		return ("no_but_other_payment", other_payment)

	return ("NOT_DETECT", Extractions())

def check_for_signup(text):
	premise_signup_success = [
		"Done",
		"I have created my account",
		"Done, I have the account"
	]
	if check_entailment(premise_signup_success, text):
		return "signup_success"

	premise_signup_fail = [
		"I can not sign up the account",
		"There is some problems with the sign up process",
		"I follow the link you gave me but it is too complicated",
		"I follow the link you gave me but still am unable to create the account"
	]
	if check_entailment(premise_signup_fail, text):
		return "signup_fail"

	return "NOT_DETECT"

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
					# Try the first payment method
					payment = get_payment_method() #Extractions
					return ({self._trigger_name: 1.0}, 0.0, payment)
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
				account = check_for_e_account(observation.text)
				if account:
					extraction = Extractions()
					extraction.add_extraction("account", account)
					return ({"account": 1.0}, 0.0, extraction)
		return ({}, 1.0, Extractions())
    
class AskPaymentTriggerDetector(TriggerDetector):
	def __init__(self, trigger_name="ask_payment_response"):
		self._trigger_name = trigger_name

	@property
	def trigger_names(self) -> List[str]:
		return ["yes_payment", "no_payment", "no_but_other_payment", "no_but_try_payment"]

	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
		for observation in observations:
			if isinstance(observation, MessageObservation) and old_extractions.has_extraction("PAYMENT"):
				response, extraction = check_for_ask_payment(observation.text, old_extractions.extraction("PAYMENT"))
				if response != "NOT_DETECT":
					if response == "no_payment":
						#extraction is next_payment
						return ({response: 1.0}, 0.0, extraction) if extraction else ({}, 1.0, Extractions()) #else: already tried every payment method
					elif response == "no_but_try_payment":
						#extraction is signup_link
						return ({response: 1.0}, 0.0, extraction)
					elif response == "no_but_other_payment":
						#extraction is other_payment
						return ({response: 1.0}, 0.0, extraction)
					else:
						return ({response: 1.0}, 0.0, Extractions())
		return ({}, 1.0, Extractions())

class SignupTriggerDetector(TriggerDetector):

	def __init__(self, trigger_name="signup"):
		self._trigger_name = trigger_name
      
	@property
	def trigger_names(self) -> List[str]:
		return ["signup_success", "signup_fail"]
    
	def trigger_probabilities(self, observations: List[Observation], old_extractions: Extractions) -> Tuple[Mapping[str, float], float, Extractions]:
		for observation in observations:
			if isinstance(observation, MessageObservation) and old_extractions.has_extraction("PAYMENT"):
				response = check_for_signup(observation.text)
				if response != "NOT_DETECT":
					if response == "signup_fail":
						next_payment = get_payment_method() #Extractions
						return ({response: 1.0}, 0.0, next_payment) if next_payment else ({}, 1.0, Extractions()) #else: already tried every payment method
					else:
						return ({response: 1.0}, 0.0, Extractions())
		return ({}, 1.0, Extractions())

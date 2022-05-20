from puppeteer import MessageObservation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

def get_entailment_score(premises, observations):
	max_score = 0
	for observation in observations:
		if isinstance(observation, MessageObservation):
			msg = observation.text
			for pm in premises:
				# print("NLI: p=\"{}\", h=\"{}\"".format(pm, msg))
				x = tokenizer.encode(pm, msg, return_tensors='pt')
				logits = nli_model(x)[0] #(contradiction, neutral, entailment)
				probs = logits.softmax(dim=1)
				# retrieve an entailment score (idx=2)
				entailment_score = probs[0][2].item()
				if entailment_score > max_score:
					max_score = entailment_score
	return max_score

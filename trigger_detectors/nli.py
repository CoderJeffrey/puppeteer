from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
import torch
torch.set_printoptions(precision=4)

def check_entailment(premises, msg):
	for pm in premises:
		# print("NLI: p=\"{}\", h=\"{}\"".format(pm, msg))
		x = tokenizer.encode(pm, msg, return_tensors='pt')
		logits = nli_model(x)[0] #(contradiction, neutral, entailment)
		probs = logits.softmax(dim=1)
		ans = torch.argmax(probs)
		# print(pm, msg, ans)
		if ans == 2: #entailment
			return True
	return False

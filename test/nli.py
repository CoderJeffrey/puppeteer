from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
torch.set_printoptions(precision=4)
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
#input
while True:
	premise = input("Enter your premise: ")
	if premise == "exit":
		break
	hypothesis = input("Enter your hypothesis: ")
	print("p: {}, h: {}".format(premise, hypothesis))
	#prediction
	x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
	logits = nli_model(x)[0] #(contradiction, neutral, entailment)
	probs = logits.softmax(dim=1)
	#print
	for y in probs:
		print("c: {:4f}, n: {:4f}, e: {:4f}".format(y[0], y[1], y[2]))
	print("0: contradict, 1: neutral, 2: entailment")
	print("answer: {}".format(torch.argmax(probs)))
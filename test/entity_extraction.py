import spacy

nlp = spacy.load("en_core_web_sm")
while True:
	sent = input("Enter your sentence: ")
	if sent == "exit":
		break
	doc = nlp(sent)
	print(doc.ents)

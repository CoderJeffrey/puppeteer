import transformers
#have to "import transformers" on the previous line otherwise the next line will raise segmentation fault error.
from sentence_transformers import SentenceTransformer, util
from itertools import combinations

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["We don't delivery this item.",
    "Sorry, we don't provide shipping.", 
    "We do offer delivery."]
pair_sentences = list(combinations(sentences, 2))
# print(pair_sentences)

for s1, s2 in pair_sentences:
	emb1 = model.encode(s1)
	emb2 = model.encode(s2)
	cos_sim = util.cos_sim(emb1, emb2)
	print("s1: {}, s2: {}".format(s1, s2))
	print("Cosine-Similarity:", cos_sim)
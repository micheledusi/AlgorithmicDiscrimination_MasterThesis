#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a brief experiment about BERT spatial distribution of embeddings.
# To allow deterministic comparison between embeddings, the context is standardized.
# Please notice that this is not how BERT should be used, but that's done only to obtain a single deterministic
# embedding for a given word / token.

import torch
from src.models.word_encoder import WordEncoder


def detect_gender_direction(enc: WordEncoder):
	w_he = "he"
	w_she = "she"

	with torch.no_grad():
		emb_he = enc.embed_word(w_he, layers=[0])
		emb_she = enc.embed_word(w_she, layers=[0])

	cos_fun = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
	similarity = cos_fun(emb_he, emb_she)

	print("Cosine similarity between <he> and <she>: ", similarity)

	gender_direction = emb_he - emb_she
	print("Single embedding size: ", gender_direction.size())


def launch() -> None:
	enc = WordEncoder()
	detect_gender_direction(enc)
	return

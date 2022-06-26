#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Using an SVM to predict gender from the embeddings
import random

import numpy as np
from sklearn import svm

import settings
from src.models.word_encoder import WordEncoder
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS


M_CLASS: int = 0
F_CLASS: int = 1

encoder_m = WordEncoder(bert_model=settings.DEFAULT_BERT_MODEL_NAME)
encoder_f = WordEncoder(bert_model=settings.DEFAULT_BERT_MODEL_NAME)
encoder_m.set_embedding_template("[CLS] he is a %s [SEP]", 4)
encoder_f.set_embedding_template("[CLS] she is a %s [SEP]", 4)


def m_embed(word: str) -> np.ndarray:
	return encoder_m.embed_word_merged(word, layers=[12]).detach().numpy()[0]


def f_embed(word: str) -> np.ndarray:
	return encoder_f.embed_word_merged(word, layers=[12]).detach().numpy()[0]


def launch() -> None:
	# Occupations
	occupations = ONEWORD_OCCUPATIONS
	random.shuffle(occupations)
	split_index = int(len(occupations) * 0.8)
	train_occs = occupations[:split_index]
	eval_occs = occupations[split_index:]

	# TRAINING
	print("Training SVM...", end='')
	train_xs = []
	train_ys = []
	for occ in train_occs:
		m_emb = m_embed(occ)
		f_emb = f_embed(occ)
		train_xs.extend([m_emb, f_emb])
		train_ys.extend([M_CLASS, F_CLASS])

	classifier = svm.SVC()
	classifier.fit(train_xs, train_ys)
	print("Completed.")

	# EVALUATION
	print("Evaluating SVM...", end='')
	eval_xs = []
	eval_ys = []
	predicted_occs = []
	for occ in eval_occs:
		m_emb = m_embed(occ)
		f_emb = f_embed(occ)
		eval_xs.extend([m_emb, f_emb])
		eval_ys.extend([M_CLASS, F_CLASS])
		predicted_occs.extend(["[M] " + occ, "[F] " + occ])

	predicted_y = classifier.predict(eval_xs)
	print("Completed.")

	print("List of prediction errors:")
	n = 0
	for ey, py, occ in zip(eval_ys, predicted_y, predicted_occs):
		if ey == py:
			n += 1
		else:
			print(f"{occ:20s} Expected: {ey}\t\tPredicted: {py}")
	print()
	print(f"Totale:      {n}/{len(predicted_occs)}")
	print(f"Percentuale: {n/len(predicted_occs)*100:5.3f}%")
	pass


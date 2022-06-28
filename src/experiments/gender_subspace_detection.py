#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a brief experiment about BERT spatial distribution of embeddings.
# To allow deterministic comparison between embeddings, the context is standardized.
# Please notice that this is not how BERT should be used, but that's done only to obtain a single deterministic
# embedding for a given word / token.

import numpy as np
import torch
from sklearn import svm

import settings
from src.models.gender_enum import Gender
from src.models.word_encoder import WordEncoder
from src.parsers.winogender_occupations_parser import OccupationsParser

gendered_words: dict[Gender, list[str]] = {
	Gender.MALE: ["he", "him", "his", "male", "boy", "man", "father", "dad", "daddy", "sir", "king"],
	Gender.FEMALE: ["she", "her", "female", "girl", "woman", "mother", "mom", "mommy", "madam", "queen"],
}

LAYER: int = 12


def detect_gender_direction(encoder: WordEncoder):

	target_words: list[str] = OccupationsParser().occupations_list

	train_x: list[np.ndarray] = []
	train_y: list[Gender] = []
	eval_x: list[np.ndarray] = []

	with torch.no_grad():
		for gend, words in gendered_words.items():
			for w in words:
				embedding = encoder.embed_word_merged(w, layers=[LAYER]).detach().numpy()[0]
				train_x.append(embedding)
				train_y.append(gend)
		for tw in target_words:
			embedding = encoder.embed_word_merged(tw, layers=[LAYER]).detach().numpy()[0]
			eval_x.append(embedding)

	print("Training Dataset   - length: ", len(train_x))
	print("Evaluation Dataset - length: ", len(eval_x))

	print("Training Support Vector Classifier...", end='')
	classifier = svm.LinearSVC()
	classifier.fit(train_x, train_y)
	print("Completed.")

	print("Predicting target words...", end='')
	predicted_y = classifier.predict(eval_x)
	coefficients = classifier.coef_[0]
	print("Completed.")

	print("Results:")
	with open(f"{settings.FOLDER_RESULTS}/gender_subspace_detection/tables/occupations_static_spectrum.tsv", "w") as f:
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(["word", "gender", "gender_name", "gender_projection"]), file=f)
		for x, y, tw in zip(eval_x, predicted_y, target_words):
			spectrum = np.dot(x, coefficients)
			print(f"{tw:20s}: {Gender(y).name:6s} : {spectrum}")
			print(tw, end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(Gender(y), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(Gender(y).name, end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(spectrum, file=f)


def launch() -> None:
	enc = WordEncoder()
	detect_gender_direction(enc)
	return

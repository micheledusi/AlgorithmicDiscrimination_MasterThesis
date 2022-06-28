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

import settings
from src.models.gender_enum import Gender
from src.models.gender_subspace_model import GenderSubspaceModel
from src.models.word_encoder import WordEncoder
from src.parsers.winogender_occupations_parser import OccupationsParser
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS


gendered_words: dict[Gender, list[str]] = {
	Gender.MALE: ["he", "him", "his", "male", "boy", "man", "father", "dad", "daddy", "sir", "king", "masculinity",
	              "lord", "uncle", "grandpa", "grandfather", "husband"],
	Gender.FEMALE: ["she", "her", "female", "girl", "woman", "mother", "mom", "mommy", "madam", "queen", "femininity",
	                "lady", "aunt", "grandma", "grandmother", "wife"],
}

LAYERS: range = range(13)


def detect_gender_direction(encoder: WordEncoder):

	# target_words: list[str] = OccupationsParser().occupations_list
	target_words: list[str] = ONEWORD_OCCUPATIONS

	train_x: list[np.ndarray] = []
	train_y: list[Gender] = []
	eval_x: list[np.ndarray] = []

	with torch.no_grad():
		for gend, words in gendered_words.items():
			for w in words:
				embedding = encoder.embed_word_merged(w, layers=LAYERS).detach().numpy()
				train_x.append(embedding)
				train_y.append(gend)
		for tw in target_words:
			embedding = encoder.embed_word_merged(tw, layers=LAYERS).detach().numpy()
			eval_x.append(embedding)

	print("Training Dataset   - length: ", len(train_x))
	print("Evaluation Dataset - length: ", len(eval_x))

	# Training the gender subspace division model
	subspace_model = GenderSubspaceModel(embeddings=np.asarray(train_x), genders=train_y)

	predicted_gender = subspace_model.predict(embeddings=np.asarray(eval_x))
	projected_gender = subspace_model.project(embeddings=np.asarray(eval_x))

	with open(f"{settings.FOLDER_RESULTS}/gender_subspace_detection/tables/occupations_static_spectrum.tsv", "w") as f:
		# Printing the header
		header_list: list[str] = ["word"]
		header_list.extend([f"pred_gender_{layer:02d}" for layer in range(subspace_model.num_layers)])
		header_list.extend([f"proj_gender_{layer:02d}" for layer in range(subspace_model.num_layers)])
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(header_list), file=f)

		# Printing the table
		for word, pred, proj in zip(target_words, predicted_gender, projected_gender):
			row_list = [word]
			row_list.extend(map(str, pred))
			row_list.extend(map(str, proj))
			print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(row_list), file=f)


def launch() -> None:
	enc = WordEncoder()
	# Detecting the gender direction
	detect_gender_direction(enc)
	return

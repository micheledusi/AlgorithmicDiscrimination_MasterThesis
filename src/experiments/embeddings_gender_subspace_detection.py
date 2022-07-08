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
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS
from src.viewers.plot_gender_subspace import GenderSubspacePlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

gendered_words: dict[Gender, list[str]] = {
	Gender.MALE: ["he", "him", "his",
	              "man", "male", "boy", "masculinity", "masculine",
	              "husband", "father", "dad", "daddy", "uncle", "grandpa", "grandfather",
	              "brother", "son", "nephew",
	              "sir", "king", "lord", "prince", "master",
	              ],
	Gender.FEMALE: ["she", "her", "her",
	                "woman", "female", "girl", "femininity", "feminine",
	                "wife", "mother", "mom", "mommy", "aunt", "grandma", "grandmother",
	                "sister", "daughter", "niece",
	                "madam", "queen", "lady", "princess", "mistress",
	                ],
}

gendered_animal_words: dict[Gender, list[str]] = {
	# Gender.NEUTER: ["rabbit", "horse", "sheep", "pig", "chicken", "duck", "cattle", "goose", "fox", "tiger", "lion", ],
	Gender.MALE: ["buck", "stallion", "raw", "boar", "rooster", "drake", "bull", "gander", "fox", "tiger", "lion", ],
	Gender.FEMALE: ["doe", "mare", "ewe", "sow", "hen", "duck", "cow", "goose", "vixen", "tigress", "lioness", ],
}

LAYERS: range = range(13)


def validate_model(model: GenderSubspaceModel, validation_x: list[np.ndarray], validation_y: list[Gender],
                   layers_labels: list[str]) -> None:
	predicted_valid_y = model.predict(embeddings=np.asarray(validation_x))
	errors_per_layer = np.zeros(shape=model.num_layers, dtype=np.uint8)
	for vy, pys in zip(validation_y, predicted_valid_y):
		errors_per_layer += [vy != py for py in pys]
	accuracy_per_layer = np.ones(shape=errors_per_layer.shape) - errors_per_layer / len(validation_y)
	print(f"Errors:     ", errors_per_layer)
	for label, acc in zip(layers_labels, accuracy_per_layer):
		print(f"Layer {label:s}: acc = {acc:6.4%}")


def detect_gender_direction(encoder: WordEncoder, layers: list[int] | range, model_id: str,
                            folder_output_images: str, folder_output_tables: str) -> None:
	"""
	In this experiment we detect the gender direction with a Linear Support Vector Classifier.
	The gender direction is the orthogonal direction to the hyperplane that best divides the considered two genders.
	The hyperplane coefficients are the ones of the trained LinearSVC.

	:param folder_output_tables: The folder where to put computed tables of results
	:param folder_output_images: The folder where to put produced images with results' plots
	:param layers: the selected layers of the model. These layers will be used in the experiment
	:param encoder: the WordEncoder model used to compute the embeddings.
	:param model_id: The identifier for the model
	:return: None
	"""

	# The words we want to analyze
	target_words: list[str] = ONEWORD_OCCUPATIONS

	train_x: list[np.ndarray] = []
	train_y: list[Gender] = []
	valid_x: list[np.ndarray] = []
	valid_y: list[Gender] = []
	eval_x: list[np.ndarray] = []

	with torch.no_grad():
		# Training set - used to train the model
		for gend, words in gendered_words.items():
			for w in words:
				embedding = encoder.embed_word_merged(w, layers=layers).cpu().detach().numpy()
				train_x.append(embedding)
				train_y.append(gend)
		# Validation set - we know the labels, so we can test the model
		for gend, words in gendered_animal_words.items():
			for w in words:
				embedding = encoder.embed_word_merged(w, layers=layers).cpu().detach().numpy()
				valid_x.append(embedding)
				valid_y.append(gend)
		# Evaluation set - the experiment set, we don't know the labels
		for tw in target_words:
			embedding = encoder.embed_word_merged(tw, layers=layers).cpu().detach().numpy()
			eval_x.append(embedding)

	print("Training Dataset   - length: ", len(train_x))
	print("Validation Dataset - length: ", len(valid_x))
	print("Evaluation Dataset - length: ", len(eval_x))

	# Training the gender subspace division model
	subspace_model = GenderSubspaceModel(embeddings=np.asarray(train_x), genders=train_y, print_summary=False)

	# Validation
	layer_indices_labels = [f"{layer:02d}" for layer in layers]
	validate_model(model=subspace_model, validation_x=valid_x, validation_y=valid_y, layers_labels=layer_indices_labels)

	# Analyze components
	subspace_plotter: GenderSubspacePlotter = GenderSubspacePlotter(model=subspace_model, layers_labels=layer_indices_labels)
	subspace_plotter.plot_maximum_coefficients(
		savepath=folder_output_images + f"/coefficients_plot_{model_id}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")
	subspace_plotter.plot_2d_gendered_scatter_embeddings(
		embeddings=np.asarray(eval_x),
		save_path=folder_output_images + f"/scatter_subspace_{model_id}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")

	# Evaluation
	predicted_gender = subspace_model.predict(embeddings=np.asarray(eval_x))
	projected_gender = subspace_model.project(embeddings=np.asarray(eval_x))

	with open(f"{folder_output_tables}/occupations_static_spectrum_{model_id}.{settings.OUTPUT_TABLE_FILE_EXTENSION}", "w") as f:
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
	detect_gender_direction(encoder=enc, model_id='base', layers=LAYERS,
	                        folder_output_images=FOLDER_OUTPUT_IMAGES, folder_output_tables=FOLDER_OUTPUT_TABLES)
	return

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
from src.models.gender_classifier import GenderLinearSupportVectorClassifier, GenderDecisionTreeClassifier
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
	              "sir", "king", "lord", "prince", "duke", "master",
	              ],
	Gender.FEMALE: ["she", "her", "her",
	                "woman", "female", "girl", "femininity", "feminine",
	                "wife", "mother", "mom", "mommy", "aunt", "grandma", "grandmother",
	                "sister", "daughter", "niece",
	                "madam", "queen", "lady", "princess", "duchess", "mistress",
	                ],
}

gendered_animal_words: dict[Gender, list[str]] = {
	# Gender.NEUTER: ["rabbit", "horse", "sheep", "pig", "chicken", "duck", "cattle", "goose", "fox", "tiger", "lion", ],
	Gender.MALE: ["buck", "stallion", "raw", "boar", "rooster", "drake", "bull", ],
	Gender.FEMALE: ["doe", "mare", "ewe", "sow", "hen", "duck", "cow", ],
}

LAYERS: range = range(13)


def get_labeled_dataset(encoder: WordEncoder, layers: list[int] | range,
                        data: dict[Gender, list[str]] | list[str]) -> tuple[list[np.ndarray], list[Gender] | None]:
	"""
	This method gets a collection of words and returns the proper dataset of embeddings and labels.
	The collection can have different types:
		- If it's a dictionary, then it must have the genders as keys, and the values must be list of words of the associated gender.
		- If it's a list, then the words are the element of the list. In this case, the dataset will not contain any label.

	In the former case, the words are associated with gender and the dataset can be labeled.
	In the latter case, the gender is unknown and an unlabeled dataset is returned.

	The method also takes the encoder used to create the embeddings and the layers passed to the encoder.

	:param encoder: The encoder object that computes the embedding of a word.
	:param layers: The layers to give to the encoder.
	:param data: The starting data.
	:return: The labeled dataset, if the information on genders is provided, or the unlabeled dataset otherwise.
	"""
	x: list[np.ndarray] = []
	y: list[Gender] = []
	with torch.no_grad():
		if isinstance(data, list):
			for w in data:
				embedding = encoder.embed_word_merged(w, layers=layers).cpu().detach().numpy()
				x.append(embedding)
			return x, None
		elif isinstance(data, dict):
			for gend, words in data.items():
				for w in words:
					embedding = encoder.embed_word_merged(w, layers=layers).cpu().detach().numpy()
					x.append(embedding)
					y.append(gend)
			return x, y
		else:
			raise AttributeError(f"Cannot convert data of type {type(data)} to a proper dataset")


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

	train_x, train_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_words)
	valid_x, valid_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_animal_words)
	eval_x,        _ = get_labeled_dataset(encoder=encoder, layers=layers, data=target_words)

	print("Training Dataset   - length: ", len(train_x))
	print("Validation Dataset - length: ", len(valid_x))
	print("Evaluation Dataset - length: ", len(eval_x))

	# Training the gender subspace division model
	layer_indices_labels = [f"{layer:02d}" for layer in layers]
	subspace_model = GenderLinearSupportVectorClassifier(training_embeddings=np.asarray(train_x),
	                                                     training_genders=train_y, layers_labels=layer_indices_labels,
	                                                     print_summary=False)
	# Validation
	subspace_model.evaluate(evaluation_embeddings=valid_x, evaluation_genders=valid_y)

	# Analyze components
	subspace_plotter: GenderSubspacePlotter = GenderSubspacePlotter(model=subspace_model,
	                                                                layers_labels=layer_indices_labels)
	subspace_plotter.plot_most_important_features(
		savepath=folder_output_images + f"/coefficients_plot_{model_id}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")
	subspace_plotter.plot_2d_gendered_scatter_embeddings(
		embeddings=np.asarray(eval_x),
		save_path=folder_output_images + f"/scatter_subspace_{model_id}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")

	# Evaluation
	predicted_gender = subspace_model.predict(embeddings=np.asarray(eval_x))
	projected_gender = subspace_model.project(embeddings=np.asarray(eval_x))

	with open(f"{folder_output_tables}/occupations_static_spectrum_{model_id}.{settings.OUTPUT_TABLE_FILE_EXTENSION}",
	          "w") as f:
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
	detect_gender_direction(encoder=enc, model_id='base-lsvc', layers=LAYERS,
	                        folder_output_images=FOLDER_OUTPUT_IMAGES, folder_output_tables=FOLDER_OUTPUT_TABLES)
	return

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
from src.models.gender_classifier import GenderLinearSupportVectorClassifier, GenderDecisionTreeClassifier, \
	_AbstractGenderClassifier
from src.models.word_encoder import WordEncoder
from src.parsers import jobs_parser
from src.viewers.plot_gender_subspace import GenderSubspacePlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

gendered_words: dict[Gender, list[str]] = {
	Gender.MALE: ["he", "him", "his",
	              "man", "male", "boy", "masculinity", "masculine", "manly",
	              "husband", "father", "dad", "daddy", "uncle", "grandpa", "grandfather",
	              "brother", "son", "nephew", "grandson", "widower",
	              "sir", "king", "lord", "prince", "duke", "master", "emperor",
	              "marquess", "earl", "viscount", "baron",
	              ],
	Gender.FEMALE: ["she", "her", "her",
	                "woman", "female", "girl", "femininity", "feminine", "womanly",
	                "wife", "mother", "mom", "mommy", "aunt", "grandma", "grandmother",
	                "sister", "daughter", "niece", "granddaughter", "widow",
	                "madam", "queen", "lady", "princess", "duchess", "mistress", "empress",
	                "marchioness", "countess", "viscountess", "baroness",
	                ],
}

gendered_animal_words: dict[Gender, list[str]] = {
	# Gender.NEUTER: ["rabbit", "horse", "sheep", "pig", "chicken", "duck", "cattle", "goose", "fox", "tiger", "lion", ],
	Gender.MALE: ["buck", "stallion", "raw", "boar", "rooster", "drake", "bull", ],
	Gender.FEMALE: ["doe", "mare", "ewe", "sow", "hen", "duck", "cow", ],
}

LAYERS: range = range(0, 13)


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


def get_trained_classifier(classifier_class, model_name: str, encoder: WordEncoder,
                           layers: list[int] | range, layers_labels: list[str]) -> _AbstractGenderClassifier:
	train_x, train_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_words)
	valid_x, valid_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_animal_words)

	# Training the gender subspace division model
	print("Training model: ", model_name)
	classifier = classifier_class(name=model_name, training_embeddings=np.asarray(train_x),
	                              training_genders=train_y, layers_labels=layers_labels, print_summary=False)

	# Validation
	print("Evaluating model: ", model_name)
	classifier.evaluate(evaluation_embeddings=valid_x, evaluation_genders=valid_y)

	return classifier


def detect_gender_direction(classifier: _AbstractGenderClassifier, encoder: WordEncoder,
                            layers: list[int] | range, layers_labels: list[str],
                            folder_output_images: str, folder_output_tables: str) -> None:
	"""
	In this experiment we detect the gender direction with a Linear Support Vector Classifier.
	The gender direction is the orthogonal direction to the hyperplane that best divides the considered two genders.
	The hyperplane coefficients are the ones of the trained LinearSVC.

	:param encoder: The word encoder
	:param layers: The layers to analyze of the testing dataset
	:param layers_labels: The labels for the given layers
	:param folder_output_tables: The folder where to put computed tables of results
	:param folder_output_images: The folder where to put produced images with results' plots
	:param classifier: The classifier model
	:return: None
	"""

	# The words we want to analyze
	target_words: list[str] = jobs_parser.get_words_list()
	eval_x, _ = get_labeled_dataset(encoder=encoder, layers=layers, data=target_words)

	# Analyze components
	subspace_plotter: GenderSubspacePlotter = GenderSubspacePlotter(model=classifier,
	                                                                layers_labels=layers_labels)
	subspace_plotter.plot_most_important_features(
		savepath=folder_output_images + f"/coefficients_plot_{classifier.name}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")
	subspace_plotter.plot_2d_gendered_scatter_embeddings(
		embeddings=np.asarray(eval_x),
		save_path=folder_output_images + f"/scatter_subspace_{classifier.name}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")
	try:
		subspace_plotter.plot_tree(
			savepath=folder_output_images + f"/tree_scheme_{classifier.name}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}")
	except NotImplementedError:
		pass

	# Evaluation
	predicted_gender_class = classifier.predict_gender_class(embeddings=np.asarray(eval_x))
	predicted_gender_spectrum = classifier.predict_gender_spectrum(embeddings=np.asarray(eval_x))
	computed_gender_relevance = classifier.compute_gender_relevance(embeddings=np.asarray(eval_x))

	with open(
			f"{folder_output_tables}/occupations_static_spectrum_{classifier.name}.{settings.OUTPUT_TABLE_FILE_EXTENSION}",
			"w") as f:
		# Printing the header
		header_list: list[str] = ["word"]
		header_list.extend([f"gender_class_{layer:02d}" for layer in range(classifier.num_layers)])
		header_list.extend([f"gender_spectrum_{layer:02d}" for layer in range(classifier.num_layers)])
		header_list.extend([f"gender_relevance_{layer:02d}" for layer in range(classifier.num_layers)])
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(header_list), file=f)

		# Printing the table
		for word, g_class, g_spectrum, g_relevance in zip(target_words, predicted_gender_class,
		                                                  predicted_gender_spectrum, computed_gender_relevance):
			row_list = [word]
			row_list.extend(map(str, g_class))
			row_list.extend(map(str, g_spectrum))
			row_list.extend(map(str, g_relevance))
			print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(row_list), file=f)


def launch() -> None:
	# Encoder
	enc = WordEncoder()
	# Layers
	layers = LAYERS
	layer_indices_labels = [f"{layer:02d}" for layer in layers]

	gender_classifier_lsvc = get_trained_classifier(GenderLinearSupportVectorClassifier, model_name="base-lsvc",
	                                                encoder=enc, layers=layers, layers_labels=layer_indices_labels)
	gender_classifier_tree = get_trained_classifier(GenderDecisionTreeClassifier, model_name="base-tree",
	                                                encoder=enc, layers=layers, layers_labels=layer_indices_labels)

	# Detecting the gender direction
	detect_gender_direction(classifier=gender_classifier_lsvc, encoder=enc,
	                        layers=layers, layers_labels=layer_indices_labels,
	                        folder_output_images=FOLDER_OUTPUT_IMAGES, folder_output_tables=FOLDER_OUTPUT_TABLES)
	detect_gender_direction(classifier=gender_classifier_tree, encoder=enc,
	                        layers=layers, layers_labels=layer_indices_labels,
	                        folder_output_images=FOLDER_OUTPUT_IMAGES, folder_output_tables=FOLDER_OUTPUT_TABLES)
	return

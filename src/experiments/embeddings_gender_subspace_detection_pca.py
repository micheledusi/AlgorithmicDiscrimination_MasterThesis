#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script emulates the Gender Subspace Detection experiment, but with fewer dimensions.
# The pipeline is:
# We take the embeddings of BERT for single words
# We project over the first N (=10 by default) features. Those features are selected depending on the relevance in the gender classification.
# We reduce those N features to 2 with PCA.
# We plot the 2 dimensions

import numpy as np
import torch
from matplotlib import pyplot as plt

import settings
from src.experiments.embeddings_gender_subspace_detection import gendered_words, get_labeled_dataset
from src.models.gender_classifier import GenderLinearSupportVectorClassifier
from src.models.word_encoder import WordEncoder
from src.parsers.serializer import Serializer
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection_pca"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

LAYERS: range = range(10, 13)
DIM_SELECTION_SIZE: int = 50


def launch() -> None:
	# Encoder
	enc = WordEncoder()
	# Layers
	layers = LAYERS
	layer_indices_labels = [f"{layer:02d}" for layer in layers]

	# Building the training dataset
	train_x, train_y = get_labeled_dataset(encoder=enc, layers=layers, data=gendered_words)

	# Training the gender subspace division model
	classifier = GenderLinearSupportVectorClassifier(name="base-lsvc", training_embeddings=np.asarray(train_x),
	                                                 training_genders=train_y, layers_labels=layer_indices_labels)
	# Extracting the most relevant features for each layer
	clf_features = classifier.get_most_important_features()
	# We take only the first N dimensions
	selected_features = [indices[:DIM_SELECTION_SIZE] for (indices, _) in clf_features]

	# Retrieving embeddings with dimensions: (1678, 3, 768)
	ser = Serializer()
	embeddings: torch.Tensor = ser.load_embeddings("jobs", 'pt')
	print("embeddings.size = ", embeddings.size())
	# We'll need to select only the wanted embeddings
	gender_spectrum = classifier.predict_gender_spectrum(embeddings.detach().numpy())
	gender_spectrum = gender_spectrum.swapaxes(0, 1)
	print("gender_spectrum.shape = ", gender_spectrum.shape)

	for layer, label, features, spectrum in zip(layers, layer_indices_labels, selected_features, gender_spectrum):
		print("Current layer: ", label)
		# indices.shape = (768,)
		# importance.shape = (768,)

		layer_reduced_embeddings = embeddings[:, layer, features]
		print(f"Layer embeddings reduced to N={DIM_SELECTION_SIZE} dimensions: ", layer_reduced_embeddings.shape)

		cmap = settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE
		# cmap = plt.get_cmap('rainbow')

		# Using plotter to visualize embeddings
		plotter = EmbeddingsScatterPlotter(layer_reduced_embeddings)
		plotter.colormap = cmap
		plotter.colors = spectrum
		plotter.sizes = 12
		ax = plotter.plot_2d_pc()
		ax.scatter([0], [0], c='k')
		plt.suptitle("Layer " + label)
		plotter.save(filename=FOLDER_OUTPUT_IMAGES + f'/reduced_to_{DIM_SELECTION_SIZE}_layer.{label}.' + settings.OUTPUT_IMAGE_FILE_EXTENSION)
		# plotter.show()
	return

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
from src.models.gender_classifier import GenderLinearSupportVectorClassifier
from src.parsers.serializer import Serializer
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection_pca"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

LAYERS: range = range(8, 13)
DIM_SELECTION_SIZE: int = 350


def launch() -> None:
	ser = Serializer()

	# Layers
	layers = LAYERS
	layer_indices_labels = [f"{layer:02d}" for layer in layers]

	# Retrieving the training dataset
	gendered_ds = ser.load_dataset('gendered_words')
	train_x = np.asarray(gendered_ds['embedding'], dtype=np.float)[:, layers]
	train_y = np.asarray(gendered_ds['gender'])
	print("Train_X shape = ", train_x.shape)
	print("Train_Y shape = ", train_y.shape)

	# Training the gender subspace division model
	classifier = GenderLinearSupportVectorClassifier(name="base-lsvc", training_embeddings=train_x,
	                                                 training_genders=train_y, layers_labels=layer_indices_labels)
	# Extracting the most relevant features for each layer
	clf_features = classifier.get_most_important_features()
	# We take only the first N dimensions
	selected_features = [indices[:DIM_SELECTION_SIZE] for (indices, _) in clf_features]

	# Retrieving embeddings with dimensions: (1678, 3, 768)
	embeddings: torch.Tensor = ser.load_embeddings("jobs", 'pt')
	print("embeddings.size = ", embeddings.size())
	# We'll need to select only the wanted embeddings
	gender_spectrum = classifier.predict_gender_spectrum(embeddings.detach().numpy())
	gender_spectrum = gender_spectrum.swapaxes(0, 1)
	print("gender_spectrum.shape = ", gender_spectrum.shape)

	# Resetting Torch random seed for repeatable PCA results
	torch.manual_seed(settings.RANDOM_SEED)

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

		"""
		# The second part of the experiment aims to compute the approximate dimensionality of gender subspace
		print("PCA")
		reduced_embeddings, eigenvalues, principal_components = torch.pca_lowrank(embeddings[:, layer], q=DIM_SELECTION_SIZE)
		print("Embeddings reduced to: ", reduced_embeddings.shape)
		print("Principal components: ", principal_components.shape, " = Transformation matrix from 768 to N")
		explained_variance = eigenvalues / sum(eigenvalues)

		interrupted: bool = False
		partial_sum: float = 0
		for i, variance in enumerate(explained_variance):
			if variance <= 0.01:
				interrupted = True
				break
			partial_sum += variance
			print(f"Component {i:3d} - explained variance = {variance:6.3%} - partial sum = {partial_sum:6.3%}")
		if interrupted:
			print("Remaining components - explained variance < 1.00%")

		print("------------\n")
		"""
	return

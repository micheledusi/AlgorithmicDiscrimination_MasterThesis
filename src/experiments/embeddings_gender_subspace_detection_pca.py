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
from src.models.dimensionality_reducer import GenderClassifierReducer, PipelineReducer, PCAReducer, TrainedPCAReducer
from src.models.gender_classifier import GenderLinearSupportVectorClassifier, GenderDecisionTreeClassifier
from src.models.layers_iterator import LayersIterator
from src.parsers.serializer import Serializer
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection_pca"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

LAYERS: range = range(8, 13)
CMAP = settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE
# Resetting Torch random seed for repeatable PCA results
torch.manual_seed(settings.RANDOM_SEED)


def launch() -> None:
	ser = Serializer()

	# Layers
	layers = LAYERS
	layer_indices_labels = [f"{layer:02d}" for layer in layers]

	# Retrieving the training dataset
	gendered_ds = ser.load_dataset('gendered_words.tsv')
	train_x = np.asarray(gendered_ds['embedding'], dtype=np.float)[:, layers]
	train_y = np.asarray(gendered_ds['gender'])
	print("Training embeddings shape: ", train_x.shape)

	# Retrieving embeddings with dimensions: (1678, 3, 768)
	embeddings: np.ndarray = ser.load_embeddings("jobs", 'np')[:, layers]
	print("Starting embeddings shape: ", embeddings.shape)

	# Gender spectrum
	gender_clf = GenderLinearSupportVectorClassifier("gender_clf", train_x, train_y, layer_indices_labels)
	gender_spectrum = gender_clf.predict_gender_spectrum(embeddings).swapaxes(0, 1)

	# Classifiers and reducers
	lsvc_clf_768 = GenderLinearSupportVectorClassifier("base-lsvc-768", train_x, train_y, layer_indices_labels)
	reducer_768_50 = GenderClassifierReducer(from_m=768, to_n=50, classifier=lsvc_clf_768)
	train_50 = reducer_768_50.reduce(train_x)
	reducer_50_2 = TrainedPCAReducer(train_50, to_n=2)

	# Instancing the final reducer object
	reducer = PipelineReducer([
		reducer_768_50,
		reducer_50_2,
	])

	# Reducing dimensions with reducer
	print("\nReducing embeddings...")
	reduced_embeddings = reducer.reduce(embeddings)
	print("Reduced embeddings shape: ", reduced_embeddings.shape)

	for layer_emb, label, spectrum in zip(LayersIterator(reduced_embeddings), layer_indices_labels, gender_spectrum):
		# Using plotter to visualize embeddings
		plotter = EmbeddingsScatterPlotter(torch.Tensor(layer_emb))
		plotter.colormap = CMAP
		plotter.colors = spectrum
		plotter.sizes = 12
		ax = plotter.plot_2d_pc()
		ax.scatter([0], [0], c='k')
		plt.suptitle("Layer " + label)
		plotter.save(
			filename=FOLDER_OUTPUT_IMAGES + f'/pipeline_layer.{label}.' + settings.OUTPUT_IMAGE_FILE_EXTENSION)
	return

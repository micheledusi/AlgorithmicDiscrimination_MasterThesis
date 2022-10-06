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
from src.models.word_encoder import WordEncoder
from src.parsers.serializer import Serializer
from src.parsers.winogender_occupations_parser import OccupationsParser
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection_pca"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

LAYERS: range = range(12, 13)
CMAP = settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE
# Resetting Torch random seed for repeatable PCA results
torch.manual_seed(settings.RANDOM_SEED)


def launch() -> None:
	ser = Serializer()

	# Layers
	layers = LAYERS
	layer_indices_labels = [f"{layer:02d}" for layer in layers]

	# Retrieving the training dataset
	gendered_ds = ser.load_dataset('gendered_words')
	train_x = np.asarray(gendered_ds['embedding'], dtype=np.float)[:, layers]
	train_y = np.asarray(gendered_ds['gender'])
	print("Training embeddings shape: ", train_x.shape)

	# Retrieving embeddings with dimensions: (1678, #layers, 768)
	# embeddings: np.ndarray = ser.load_embeddings("jobs", 'np')[:, layers]
	# The next piece uses WinoGender occupations
	# If you want JNeidel occupations, uncomment the line above
	jobs_list: list[str] = OccupationsParser().occupations_list
	encoder: WordEncoder = WordEncoder()
	embeddings_list: list[np.ndarray] = [encoder.embed_word_merged(w, layers).cpu().detach().numpy() for w in jobs_list]
	embeddings: np.ndarray = np.stack(embeddings_list, axis=0)
	print("Starting embeddings shape: ", embeddings.shape)

	# Gender spectrum
	gender_clf = GenderLinearSupportVectorClassifier("gender_clf", train_x, train_y, layer_indices_labels)
	gender_spectrum = gender_clf.predict_gender_spectrum(embeddings).swapaxes(0, 1)

	# Classifiers and reducers
	# steps = [10, 20, 50, 75, 100, 150, 200, 250, 300, 350, 400, 500, 700]
	steps = [200]

	for step_n in steps:
		lsvc_clf_768 = GenderLinearSupportVectorClassifier("base-lsvc-768", train_x, train_y, layer_indices_labels)
		reducer_768_n = GenderClassifierReducer(from_m=768, to_n=step_n, classifier=lsvc_clf_768)
		# train_n = reducer_768_n.reduce(train_x)
		# reducer_n_2 = TrainedPCAReducer(train_n, to_n=2)
		reducer_n_2 = PCAReducer(from_m=step_n, to_n=2)

		# Instancing the final reducer object
		reducer = PipelineReducer([
			reducer_768_n,
			reducer_n_2,
		])

		# Reducing dimensions with reducer
		print("\nReducing embeddings...")
		reduced_embeddings = reducer.reduce(embeddings)
		print("Reduced embeddings shape: ", reduced_embeddings.shape)
		print()

		# Computing transformation matrix
		transformation_matrix = reducer.get_transformation_matrix()
		print("Transformation matrix: ", np.asarray(transformation_matrix).shape)
		# tm_l08 = transformation_matrix[0]
		# print("TM for layer 08: ", tm_l08)

		for layer_emb, label, spectrum in zip(LayersIterator(reduced_embeddings), layer_indices_labels, gender_spectrum):
			# Using plotter to visualize embeddings
			plotter = EmbeddingsScatterPlotter(torch.Tensor(layer_emb))
			plotter.colormap = CMAP
			plotter.colors = spectrum
			plotter.sizes = 12
			ax = plotter.plot_2d_pc()
			ax.scatter([0], [0], c='k')
			# plt.suptitle("Layer " + label)
			plotter.save(filename=FOLDER_OUTPUT_IMAGES + f'/pipeline_supervised_n_{step_n}.layer_{label}.' + settings.OUTPUT_IMAGE_FILE_EXTENSION)
	return

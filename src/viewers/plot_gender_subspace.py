#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Plotting functions for "gender subspace detection" task

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree

import settings
from src.models.gender_classifier import _AbstractGenderClassifier, GenderDecisionTreeClassifier


class GenderSubspacePlotter:

	def __init__(self, model: _AbstractGenderClassifier, layers: list[int] | range = 'all', layers_labels: list = None):
		"""
		Instance an object of class GenderSubspacePlotter.

		:param model: The "GenderSubspaceModel" instance that is taken as reference to plot. This model will predict the
		values for the embeddings, when these will be plotted.
		:param layers: A selection of layers to plot between the layers of the model. All the layers in this list/range
		MUST be contained (as indices) in the model layers. By default, all the layers will be considered.
		:param layers_labels: The names for the selected layers. If present,
		"""
		self.__model: _AbstractGenderClassifier = model
		# Sorting model coefficients
		# NOTE: WE ASSUME THE MODEL IS ALREADY TRAINED
		self.__sorted_coefs_indices: np.ndarray = np.argsort(-np.abs(self.model.features_importance), axis=-1)
		# The sorted coefficients indices are an array of [# layer, # indices].
		# For each layer, the indices are sorted from the max to the min value of the coefficients of that layer.

		# We work only on some layers
		if layers == 'all':
			self.__layers: list[int] = list(range(self.model.num_layers))
		else:
			for layer in layers:
				if layer >= self.model.num_layers:
					raise IndexError(f"Cannot select a layer with index {layer} in a model with only {self.model.num_layers} layers.")
				if layer < 0:
					raise IndexError(f"Cannot select a layer with negative index {layer}")
			self.__layers: list[int] = list(layers)
		if layers_labels is not None:
			self.__layers_labels: list = layers_labels
			assert len(self.__layers_labels) == len(self.__layers)

	@property
	def model(self) -> _AbstractGenderClassifier:
		return self.__model

	def plot_2d_gendered_scatter_embeddings(self, save_path: str, embeddings: np.ndarray) -> None:
		projections: np.ndarray = self.model.predict_gender_spectrum(embeddings)
		for layer, label in zip(self.__layers, self.__layers_labels):

			# Pre-processing the data
			layer_principal_components = self.__sorted_coefs_indices[layer, :2]
			# print(f"Layer {layer} principal components: ", layer_principal_components)
			layer_embeddings: np.ndarray = embeddings[:, layer]
			# print("Layer embeddings - shape: ", layer_embeddings.shape)
			reduced_embeddings: np.ndarray = layer_embeddings[:, layer_principal_components]
			# print("Reduced embeddings - shape: ", reduced_embeddings.shape)
			layer_projections: np.ndarray = projections[:, layer]
			# print("Layer projections - shape: ", layer_projections.shape)

			# Computing the gender direction in 2D
			reduced_gender_direction: np.ndarray = self.model.features_bias[layer, layer_principal_components]
			reduced_gender_direction = reduced_gender_direction / (np.linalg.norm(reduced_gender_direction) + 1e-16)
			gender_arrow = np.asarray([-reduced_gender_direction, reduced_gender_direction])
			# print(f"Layer {layer} reduced gender direction: ", reduced_gender_direction)

			# Plotting
			fig, ax = plt.subplots(nrows=1, ncols=1, dpi=100, figsize=(10, 8))
			ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=layer_projections, cmap=settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE)
			ax.plot(gender_arrow[:, 0], gender_arrow[:, 1], color='#000', linewidth=1.0)  # Gender direction
			ax.plot([0], [0], 'ok')  # Origin
			ax.set_title(f"Layer {label}")
			ax.set_xlabel(f"Feature {layer_principal_components[0]}")
			ax.set_ylabel(f"Feature {layer_principal_components[1]}")
			ax.plot()
			plt.savefig(save_path.replace(settings.OUTPUT_IMAGE_FILE_EXTENSION, f"{label}." + settings.OUTPUT_IMAGE_FILE_EXTENSION))
			# plt.show()
			plt.close(fig)
		return

	def plot_most_important_features(self, savepath: str, highlights: int = 10, annotations: int = 2) -> None:
		"""
		Plots the stem graphs for the SVC in the GenderSubspaceModel. With this function, the main components of the
		coefficient vectors of the model will be easily identified.
		:param savepath: The saving path for the produced image with the plot
		:param highlights: The number of the highest values that will be colored in red in every plot
		:param annotations: The number of the highest values that will have an associated label with the index and value
		:return: None
		"""
		fig, axs = plt.subplots(nrows=len(self.__layers), ncols=1, sharex='all', sharey='all',
		                        figsize=(18, 20), dpi=100, constrained_layout=True)

		for layer in self.__layers:
			layer_imprt = self.model.features_importance[layer]
			max_coefs_indices = self.__sorted_coefs_indices[layer, :highlights]
			min_coefs_indices = self.__sorted_coefs_indices[layer, highlights:]
			ann_coefs_indices = self.__sorted_coefs_indices[layer, :annotations]

			# MIN
			markerline, stemline, baseline, = axs[layer].stem(min_coefs_indices, layer_imprt[min_coefs_indices])
			plt.setp(markerline, markersize=0.5, color='#CCC')
			plt.setp(stemline, linewidth=1.0, color='#CCC')
			plt.setp(baseline, linewidth=0.5, color='k')
			# MAX
			markerline, stemline, baseline, = axs[layer].stem(max_coefs_indices, layer_imprt[max_coefs_indices])
			plt.setp(markerline, markersize=1.0, color="red")
			plt.setp(stemline, linewidth=1.0, color="red")
			plt.setp(baseline, linewidth=0.5, color='k')
			# Annotations
			for idx in ann_coefs_indices:
				axs[layer].annotate(f"{idx}: {layer_imprt[idx]:.3f}",
				                    xy=(idx, layer_imprt[idx]), xytext=(1, 0),
				                    textcoords="offset points", fontsize=8, zorder=10)

		plt.savefig(savepath)
		# plt.show()
		plt.close(fig)
		return

	def plot_tree(self, savepath: str) -> None:
		if not isinstance(self.model, GenderDecisionTreeClassifier):
			raise NotImplementedError("Cannot apply this method if the classifier is not an object of class GenderDecisionTreeClassifier")
		# Otherwise
		for layer, label in zip(self.__layers, self.__layers_labels):

			clf = self.model.classifiers[layer]
			tree.plot_tree(clf)
			plt.suptitle("Layer " + label)
			plt.savefig(savepath.replace(settings.OUTPUT_IMAGE_FILE_EXTENSION, f"{label}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}"))
			plt.close()

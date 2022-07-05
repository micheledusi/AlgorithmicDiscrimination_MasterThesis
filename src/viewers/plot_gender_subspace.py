#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Plotting functions for "gender subspace detection" task

import numpy as np
from matplotlib import pyplot as plt

import settings
from src.models.gender_subspace_model import GenderSubspaceModel


class GenderSubspacePlotter:

	__model: GenderSubspaceModel = None
	__sorted_coefs_indices: np.ndarray = None
	__layers: list[int]

	def __init__(self, model: GenderSubspaceModel, layers: list[int] | range = 'all'):
		self.__model = model
		# Sorting model coefficients
		# NOTE: WE ASSUME THE MODEL IS ALREADY TRAINED
		self.__sorted_coefs_indices = np.argsort(-np.abs(self.model.coefficients), axis=-1)
		# The sorted coefficients indices are an array of [# layer, # indices].
		# For each layer, the indices are sorted from the max to the min value of the coefficients of that layer.

		# We work only on some layers
		if layers == 'all':
			self.__layers = list(range(self.model.num_layers))
		else:
			for layer in layers:
				if layer >= self.model.num_layers:
					raise IndexError(f"Cannot select a layer with index {layer} in a model with only {self.model.num_layers} layers.")
				if layer < 0:
					raise IndexError(f"Cannot select a layer with negative index {layer}")
			self.__layers = list(layers)

	@property
	def model(self) -> GenderSubspaceModel:
		return self.__model

	def plot_2d_gendered_scatter_embeddings(self, savepath: str, embeddings: np.ndarray) -> None:
		projections: np.ndarray = self.model.project(embeddings)
		for layer in self.__layers:

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
			reduced_gender_direction: np.ndarray = self.model.coefficients[layer, layer_principal_components]
			reduced_gender_direction = reduced_gender_direction / (np.linalg.norm(reduced_gender_direction) + 1e-16)
			gender_arrow = np.asarray([-reduced_gender_direction, reduced_gender_direction])
			# print(f"Layer {layer} reduced gender direction: ", reduced_gender_direction)

			# Plotting
			fig, ax = plt.subplots(nrows=1, ncols=1, dpi=100, figsize=(10, 8))
			ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=layer_projections, cmap=settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE)
			ax.plot(gender_arrow[:, 0], gender_arrow[:, 1], color='#000', linewidth=1.0)  # Gender direction
			ax.plot([0], [0], 'ok')  # Origin
			ax.set_title(f"Layer {layer:02d}")
			ax.set_xlabel(f"Feature {layer_principal_components[0]}")
			ax.set_ylabel(f"Feature {layer_principal_components[1]}")
			ax.plot()
			plt.savefig(savepath.replace(settings.OUTPUT_IMAGE_FILE_EXTENSION, f"{layer:02d}." + settings.OUTPUT_IMAGE_FILE_EXTENSION))
			# plt.show()
		return

	def plot_maximum_coefficients(self, savepath: str, highlights: int = 10, annotations: int = 2) -> None:
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
			layer_coefs = self.model.coefficients[layer]
			max_coefs_indices = self.__sorted_coefs_indices[layer, :highlights]
			min_coefs_indices = self.__sorted_coefs_indices[layer, highlights:]
			ann_coefs_indices = self.__sorted_coefs_indices[layer, :annotations]

			# MIN
			markerline, stemline, baseline, = axs[layer].stem(min_coefs_indices, layer_coefs[min_coefs_indices])
			plt.setp(markerline, markersize=0.5, color='#CCC')
			plt.setp(stemline, linewidth=1.0, color='#CCC')
			plt.setp(baseline, linewidth=0.5, color='k')
			# MAX
			markerline, stemline, baseline, = axs[layer].stem(max_coefs_indices, layer_coefs[max_coefs_indices])
			plt.setp(markerline, markersize=1.0, color="red")
			plt.setp(stemline, linewidth=1.0, color="red")
			plt.setp(baseline, linewidth=0.5, color='k')
			# Annotations
			for idx in ann_coefs_indices:
				axs[layer].annotate(f"{idx}: {layer_coefs[idx]:.3f}",
				                    xy=(idx, layer_coefs[idx]), xytext=(1, 0),
				                    textcoords="offset points", fontsize=8, zorder=10)

		plt.savefig(savepath)
		# plt.show()
		return

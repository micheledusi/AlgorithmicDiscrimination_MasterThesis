#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# A class to reduce embeddings dimensions from M to N, according to different criteria.

from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.models.gender_classifier import _AbstractGenderClassifier
from src.models.layers_iterator import LayersIterator


class BaseDimensionalityReducer(ABC):
	"""
	The Dimensionality Reducer takes a tensor with some features and reduces this number according to various criteria.
	"""

	def __init__(self, from_m: int, to_n: int):
		"""
		Initializer for the reducer class.

		:param from_m: The number of features of the input embeddings.
		:param to_n: The number of features of the output embeddings.
		"""
		self.m = from_m
		self.n = to_n

	@staticmethod
	def _count_features(embeddings: np.ndarray | torch.Tensor) -> int:
		if isinstance(embeddings, np.ndarray):
			return embeddings.shape[-1]
		elif isinstance(embeddings, torch.Tensor):
			return embeddings.size(dim=-1)
		else:
			raise AttributeError("Cannot count last dimension of object with type: ", type(embeddings))

	@staticmethod
	def __prepare_input(embeddings: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
		if isinstance(embeddings, np.ndarray):
			embeddings = np.squeeze(embeddings)
		elif isinstance(embeddings, torch.Tensor):
			embeddings = torch.squeeze(embeddings)
		return embeddings

	def __check_input(self, embeddings: np.ndarray | torch.Tensor) -> None:
		assert self._count_features(embeddings) == self.m

	def __check_output(self, embeddings: np.ndarray | torch.Tensor) -> None:
		assert self._count_features(embeddings) == self.n

	"""
	@staticmethod
	def _to_ndarray(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
		if isinstance(embeddings, torch.Tensor):
			return embeddings.detach().numpy()
		return embeddings

	@staticmethod
	def _to_tensor(embeddings: np.ndarray | torch.Tensor) -> torch.Tensor:
		if isinstance(embeddings, np.ndarray):
			return torch.Tensor(embeddings)
		return embeddings
	"""

	def reduce(self, embeddings: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
		"""
		Applies the transformation of dimensions reduction, along the features' axis.
		The features' axis will go from length M to length N.
		:param embeddings: The input tensor, of dimensions [ d1, d2, ..., dk, M ]
		:return: The output tensor, of dimensions [ d1, d2, ..., dk, N ]
		"""
		print(f"Reducing features from M = {self.m:3d} to N = {self.n:3d} with: ", type(self))
		embeddings = self.__prepare_input(embeddings)
		self.__check_input(embeddings)
		results = self._reduction_transformation(embeddings)
		self.__check_output(results)
		return results

	@abstractmethod
	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		pass


class SelectorReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made by taking the features with specified indices.
	"""

	def __init__(self, from_m: int, indices: list[int]):
		super().__init__(from_m, len(indices))
		self._selected_features = indices

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		return np.take_along_axis(embeddings, indices=self._selected_features, axis=-1)


class MatrixReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made by multiplying the embedding tensor by a MxN matrix.
	"""

	def __init__(self, matrix: np.ndarray):
		from_m, to_n = matrix.shape
		super().__init__(from_m, to_n)
		self.__matrix = matrix

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		return np.matmul(embeddings, self.__matrix)


class PCAReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made with dynamic PCA.
	The effect of PCA depends on the given input, and it has no memory of previous results.
	For a "trained" PCA, please use 'TrainedPCAReducer'.
	"""

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		it = LayersIterator(embeddings)
		results = np.zeros(shape=(it.num_samples, it.num_layers, self.n))
		for layer_emb in it:
			pca = PCA(n_components=self.n)
			pca.fit(layer_emb)
			results[:, it.current_layer_index] = pca.transform(layer_emb)
		return results


class TrainedPCAReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class is made with PCA.
	The PCA transformation matrix is computed at first, based on the training set of the initializer.
	Then, in the reduction method, the PCA is applied with the pre-computed matrix.
	"""

	def __init__(self, train_x: np.ndarray | torch.Tensor, to_n: int):
		super().__init__(self._count_features(train_x), to_n)
		self.__pca_list: list[PCA] = []
		for layer_emb in LayersIterator(train_x):
			pca = PCA(n_components=self.n)
			pca.fit(layer_emb)
			self.__pca_list.append(pca)

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		it = LayersIterator(embeddings)
		results = np.zeros(shape=(it.num_samples, it.num_layers, self.n))
		for pca, layer_emb in zip(self.__pca_list, it):
			results[:, it.current_layer_index] = pca.transform(layer_emb)
		return results


class GenderClassifierReducer(BaseDimensionalityReducer):
	"""
	The reduction of this class uses a Gender Classifier object.
	The GenderClassifier object can be trained over a dataset (generally speaking, a gender-labeled dataset) and
	can signal the most important features over the embeddings' features.

	This reducer takes a trained classifier, extract a pre-defined number of (sorted) relevant features,
	and finally selects only those features in the input embeddings.
	"""

	def __init__(self, from_m: int, to_n: int, classifier: _AbstractGenderClassifier):
		super().__init__(from_m, to_n)
		self.__selected_features: list[np.ndarray] = []
		# Select the n most important features FOR EACH LAYER
		# (assuming n is less than the classifier features dimension)
		for (indices, _) in classifier.get_most_important_features(cut_zeros=False):
			self.__selected_features.append(indices[:self.n])

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		it = LayersIterator(embeddings)
		results = np.zeros(shape=(it.num_samples, it.num_layers, self.n))
		for indices, layer_emb in zip(self.__selected_features, it):
			results[:, it.current_layer_index] = layer_emb[:, indices]
		return results


class PipelineReducer(BaseDimensionalityReducer):
	"""
	This reducer aggregates multiple sub-reducers.
	The reducers must be given in the correct order, with compatible sizes.
	"""

	def __init__(self, reducers: list[BaseDimensionalityReducer]):
		for i in range(1, len(reducers)):
			assert reducers[i-1].n == reducers[i].m
		self.__reducers = reducers
		super().__init__(from_m=reducers[0].m, to_n=reducers[-1].n)

	def _reduction_transformation(self, embeddings: np.ndarray) -> np.ndarray:
		results: np.ndarray = embeddings
		for red in self.__reducers:
			print("\t> ", end='')
			results = red.reduce(results)
		return results




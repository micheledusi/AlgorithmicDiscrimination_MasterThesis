#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This model tries to define a gender subspace in the embeddings space, for each layers.
# For the sake of simplicity and analysis, we'll consider only male and female genders, which
# is not a socially correct assumption.

from enum import IntEnum

import numpy as np
from sklearn import svm


class GenderSubspaceModel:
	__classifiers: list[svm.LinearSVC] = []
	__num_features: int

	def __init__(self, embeddings: np.ndarray, genders: list[IntEnum] | list[int], print_summary: bool = False) -> None:
		"""
		Creates and train the hyperplane that separates genders.
		:param embeddings: A 3D matrix of embeddings: [# samples, # layers = 13, # features = 768]
		:param genders: A list of [# samples] gender, for the embeddings
		"""
		num_samples, num_layers, self.__num_features = embeddings.shape
		assert num_samples == len(genders)
		if print_summary:
			print("Training samples: ", num_samples)
			print("Number of layers: ", num_layers)
			print("Number of features: ", self.__num_features)

		# Training a classifier for each layer
		for layer in range(num_layers):
			clf = svm.LinearSVC(dual=False)
			clf.fit(embeddings[:, layer], genders)
			self.__classifiers.append(clf)

	@property
	def num_layers(self) -> int:
		return len(self.__classifiers)

	@property
	def coefficients(self) -> np.ndarray:
		return np.asarray([clf.coef_[0] for clf in self.__classifiers])

	def predict(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method predicts the gender of the embeddings, according to the trained LinearSVCs.
		Each embedding is considered as a group of #layers (=13) distinct embeddings. Each embedding will be
		processed by the layer-corresponding SVC.

		:param embeddings: A numpy array of dimensions [# samples, # layers (= 13), # features (= 768)]
		:return: A numpy array of predictions for each sample and for each layer. The array has dimensions [# samples, # layers]
		The predictions follows the class in the Gender Enumeration.
		"""
		predictions = np.zeros(shape=(len(embeddings), self.num_layers), dtype=np.uint8)
		for layer, clf in enumerate(self.__classifiers):
			predictions[:, layer] = clf.predict(embeddings[:, layer])
		return predictions

	def project(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method computes the "gender projection" of the embeddings, according to the layers LinearSVCs.
		The gender projection is:
		-   geometrically, the ratio between the projection of the embedding over the gender direction, and the gender
			direction itself.
		-   algebraically, the scalar product between the embedding and the gender direction.
		The gender direction corresponds to the coefficients of the trained SVC.
		Each embedding is considered as a group of #layers (=13) distinct embeddings. Each embedding will be
		processed by the layer-corresponding SVC.

		:param embeddings: A numpy array of dimensions [# samples, # layers (= 13), # features (= 768)]
		:return: A numpy array of projections for each sample and for each layer. The array has dimensions [# samples, # layers]
		"""
		projections = np.zeros(shape=(len(embeddings), self.num_layers), dtype=np.float)
		for layer, clf in enumerate(self.__classifiers):
			coefs = clf.coef_[0]
			projections[:, layer] = np.dot(embeddings[:, layer], coefs)
		return projections




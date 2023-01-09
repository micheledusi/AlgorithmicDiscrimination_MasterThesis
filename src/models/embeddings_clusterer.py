#########################################################################
#                            Dusi's Ph.D.                               #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This file offers some basic implementations of the clustering algorithms.
# It's specifically designed for embeddings, i.e. it takes a tensor of size
#   [# samples, # features]
# And it returns a tensor of labels, with size [# samples]

from abc import ABC, abstractmethod
from typing import final

import torch
from kmeans_pytorch import kmeans
from sklearn.cluster import AgglomerativeClustering


class ClusteringAlgorithm(ABC):

	def __init__(self, name: str, deterministic: bool):
		"""
		Initializer for the Clustering Algorithm class.
		It requires an identifier name and a flag indicating whether the algorithm is deterministic or not.
		An algorithm is deterministic when, for a specific input, it always returns the same output (i.e. it does
		not contain random components or random number generations).

		:param name: (str) The name identifier
		:param deterministic: (bool) True if it's deterministic, False otherwise.
		"""
		self.__name: str = name
		self.__deterministic: bool = deterministic

	@property
	def name(self) -> str:
		return self.__name

	@property
	def deterministic(self) -> bool:
		return self.__deterministic

	@final
	def __call__(self, samples: torch.Tensor) -> torch.Tensor:
		"""
		Calls the algorithm.
		This method wraps the class method "run", which executes the algorithm.
		This is a final method, i.e. it cannot be overwritten by the derived classes.

		:param samples: The input values, as a Tensor of size [# samples, # features]
		:return: the label of the clusters, for each input sample, with a Tensor of size [# samples]
		"""
		return self.run(samples)

	@abstractmethod
	def run(self, samples: torch.Tensor) -> torch.Tensor:
		"""
		Executes the clustering algorithm, taking the input samples and returning the labels of the clusters they belong to.
		The clusters are computed according to the logic of the class; more info on the class documentation.

		:param samples: The input values, as a Tensor of size [# samples, # features]
		:return: the label of the clusters, for each input sample, with a Tensor of size [# samples]
		"""
		pass

	@abstractmethod
	def __str__(self) -> str:
		"""
		:return: A string description of the algorithm, complete with parameters values.
		"""
		pass


class KMeansClusteringAlgorithm(ClusteringAlgorithm):

	DEFAULT_DISTANCE: str = 'cosine'

	def __init__(self, num_clusters: int = 1, distance: str = DEFAULT_DISTANCE):
		"""
		Initializes the k-Means clustering algorithm.
		This algorithm uses the SKLearn library method "kmeans" to cluster the embeddings.

		:param num_clusters: (int) The number of clusters this algorithm will find.
		:param distance: (str) 'cosine' or 'euclidean'. (Default: 'cosine').
		"""
		super().__init__("k-means", False)
		self.__num_clusters: int = num_clusters
		self.__distance: str = distance

	def run(self, samples: torch.Tensor) -> torch.Tensor:
		# Clustering with the SKLearns library
		cluster_ids_x, cluster_centers = kmeans(
			X=samples, num_clusters=self.__num_clusters,
			distance='cosine',  # Can be 'euclidean' (default) or 'cosine'
			# device=torch.device('cuda:0')
		)
		return cluster_ids_x

	def __str__(self) -> str:
		return self.name + f" (n_clusters = {self.__num_clusters}, distance = {self.__distance})"


class HierarchicalClusteringAlgorithm(ClusteringAlgorithm):

	DEFAULT_DISTANCE: str = 'cosine'
	DEFAULT_LINKAGE: str = 'average'

	def __init__(self, num_clusters: int = 1, distance: str = DEFAULT_DISTANCE, linkage: str = DEFAULT_LINKAGE):
		"""
		Initializes the hierarchical clustering algorithm.
		This algorithm uses the AgglomerativeClustering library of SKLearn to cluster the embeddings.

		:param num_clusters: (int) The number of clusters this algorithm will find.
		:param distance: (str) can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. (Default: 'cosine').
		:param linkage: (str) can be "single", "complete", "average" or "ward". (Default: 'average').
			If linkage is 'ward', distance must be 'euclidean'.
		"""
		super().__init__("hierarchical", True)
		self.__num_clusters: int = num_clusters
		self.__distance: str = distance
		self.__linkage: str = linkage

	def run(self, samples: torch.Tensor) -> torch.Tensor:
		clustering_result = AgglomerativeClustering(n_clusters=self.__num_clusters, affinity=self.__distance, linkage=self.__linkage)\
			.fit(samples.cpu().detach().numpy())
		return torch.Tensor(clustering_result.labels_)

	def __str__(self) -> str:
		return self.name + f" (n_clusters = {self.__num_clusters}, distance = {self.__distance}, linkage = {self.__linkage})"

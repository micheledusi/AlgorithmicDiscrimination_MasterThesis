#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This file offers functions to compute metrics (distances and similarities) between embeddings.

import torch
from torch import Tensor
from typing import Callable


class Metric:
	"""
	This class represents a generic metric computed over word embeddings. Metrics comprehend both distances
	(difference measures) and similarities (equality measures).

	A metric consists in:
	 * A unique name
	 * Some identifiers for the embeddings to consider
	 * A function to evaluate the metric over the considered embeddings

	The embeddings identifiers must be present as keys in the dictionary of embeddings. The dictionary of embeddings is
	given as input to the evaluating function.
	This is due in order to pass to all the metric the same dictionary and to leave to the metric itself the job to
	extract the correct embeddings.

	The evaluating function works on the *last* dimension of embeddings. This means that, if we want to compute a metric
	over tensors of dimensions:
	 `[# dim0, # dim2, # dim2, ..., # dimN, # features]`
	The result will be of shape:
	 `[# dim0, # dim2, # dim2, ..., # dimN]`
	(Supposing the evaluating functions reduces only the last dimension.
	If the programmer wants to implement something different, is their responsibility to make things work).
	"""

	def __init__(self, name: str, computational_function: Callable[[list[Tensor]], Tensor], ids: list[str]):
		self.__name = name
		self.__ids = ids
		self.__arity = len(ids)
		self.__fun = computational_function

	def __get_args(self, embeddings: dict[str, Tensor]) -> list[Tensor]:
		"""
		Gets a dictionary and returns a list of the required embeddings.
		:param embeddings: A dictionary of (keys, embeddings).
		:return: The list of arguments, as PyTorch tensors
		"""
		args = [embeddings[i] for i in self.__ids]
		assert len(args) == self.__arity
		return args

	def __call__(self, embeddings: dict[str, Tensor]) -> Tensor:
		"""
		The principal method of this class: takes the embedding dictionary, extracts the required tensors
		by its ids, and evaluate the measuring function.
		:param embeddings: A dictionary of strings and tensors
		:return: The evaluated tensor of the metric
		"""
		args = self.__get_args(embeddings)
		return self.__fun(args)

	def __str__(self) -> str:
		"""
		Returns a string describing the metric.
		This string is the concatenation of the name with the list of ids.
		:return: The representing string
		"""
		suffix: str = '_'.join(self.__ids)
		return f"{self.__name}_{suffix}"


def eucl_dist(x: Tensor, y: Tensor) -> Tensor:
	"""
	Computes the euclidean distance over two vector, along the last dimension.
	:param x: The first vector as a :class:`Tensor`
	:param y: The first vector as a :class:`Tensor`
	:return: The euclidean distance measure, as a Tensor
	"""
	return (x - y).pow(exponent=2).sum(axis=-1).sqrt()


def eucl_center_dist(points: list[Tensor]) -> Tensor:
	"""
	Computes the sum of euclidean distances of points from the center of them.
	The euclidean distance is computed along the last dimension.
	:param points: The vectors list as a list of :class:`Tensor`
	:return: The computed distance
	"""
	center = torch.mean(torch.stack(points), dim=0)
	cent_dists = [eucl_dist(x, center) for x in points]
	return torch.sum(torch.stack(cent_dists), dim=0)


def cos_simil(x: Tensor, y: Tensor) -> Tensor:
	"""
	Computes the cosine similarity between two tensors, i.e. the cosine of the angle between them.
	:param x: The first vector as a :class:`Tensor`
	:param y: The first vector as a :class:`Tensor`
	:return: The cosine similarity measure, as a Tensor
	"""
	return torch.nn.CosineSimilarity(dim=-1)(x, y)


class PairEuclideanDistance(Metric):
	"""
	Computes the euclidean distance between two vectors.
	"""

	def __init__(self, id0: str, id1: str):
		super().__init__(name="pair_eucl_dist",
		                 computational_function=lambda embs: eucl_dist(embs[0], embs[1]),
		                 ids=[id0, id1])


class TripleEuclideanDistance(Metric):
	"""
	Computes the euclidean distance between three vectors.
	We define the euclidean distance between three vectors as the sum of the three pairs of vectors.
	"""

	def __init__(self, id0: str, id1: str, id2: str):
		super().__init__(name="triple_eucl_dist",
		                 computational_function=lambda embs:
		                 eucl_dist(embs[0], embs[1]) + eucl_dist(embs[1], embs[2]) + eucl_dist(embs[2], embs[0]),
		                 ids=[id0, id1, id2])


class EuclideanCenterDistance(Metric):
	"""
	Computes the euclidean distance between a list of vectors and the center of them
	First it computes the center of the list of points by averaging their coords.
	Then, it takes all the distances of all points from the center and adds them.
	This is a generalization of the euclidean distance between a couple of points.
	"""

	def __init__(self, *ids: str):
		super().__init__(name="eucl_center_dist",
		                 computational_function=eucl_center_dist,
		                 ids=list(ids))


class PairCosineSimilarity(Metric):
	"""
	Computes the cosine similariry between two vectors, i.e. the cosine of the angle between the two.
	"""

	def __init__(self, id0: str, id1: str):
		super().__init__(name="pair_cos_simil",
		                 computational_function=lambda embs: cos_simil(embs[0], embs[1]),
		                 ids=[id0, id1])


class TripleCosineSimilarity(Metric):
	"""
	Computes the cosine similarity between three vectors.
	Cosine similarity is a similarity metric giving the angle between two vectors, i.e. not considering the length of
	the vectors. In this metric we compute the cos_simil between every pair of vectors, and multiply them together.
	"""

	def __init__(self, id0: str, id1: str, id2: str):
		super().__init__(name="triple_cos_simil",
		                 computational_function=lambda embs:
		                 torch.mul(torch.mul(cos_simil(embs[0], embs[1]), cos_simil(embs[1], embs[2])),
		                           cos_simil(embs[2], embs[0])),
		                 ids=[id0, id1, id2])


DEFAULT_SEPARATOR: str = ';'


class EmbeddingsComparator:
	"""
	This class implements a list of metrics to be computed on a set of embeddings.
	Every metric (e.g. a distance or a similarity function) is an object of (super)class :class:`Metric`.
	The embeddings must be a dictionary indexed by id-keys with list of tensors associated to them.
	"""

	def __init__(self, merge_function: Callable[[list[Tensor]], Tensor] = None):
		"""
		Initializes a Comparator object with a specific merge function.
		When the comparator is called on an embeddings dictionary, it should know how to merge different embeddings
		within the same list associated with a key. The merge functions does exactly that.
		:param merge_function: A callable parameter that takes a list of tensors and returns a single tensor. In order
			to work properly with the Metrics system, the tensors in the input list and the output tensor MUST be of
			dimensions [# layers, # features] (thus, a 2D matrix).
		"""
		# Defining a default function
		if merge_function is None:
			def merge_by_mean(tensors_list: [list[Tensor]]) -> Tensor:
				tensors_stack: Tensor = torch.stack(tensors_list)
				return torch.mean(tensors_stack, dim=0)
			merge_function = merge_by_mean
		self.__merge_function = merge_function
		self.__metrics: list[Metric] = []
		return

	def add_metric(self, metric: Metric) -> None:
		"""
		Appends a metric to the inner list of computable metrics.
		:param metric: A metrics from a :class:`Metric` subclass.
		:return: None.
		"""
		self.__metrics.append(metric)
		return

	def names_list(self) -> list[str]:
		"""
		:return: The list of names of metrics.
		"""
		return [metric.__str__() for metric in self.__metrics]

	def names_header(self, separator: str = DEFAULT_SEPARATOR) -> str:
		"""
		Returns a properly formatted header for CSV and TSV files where the list of metrics names is required.
		:param separator: A separator string interleaving the names of the metrics.
		:return: A string containing the concatenation of metrics names, interleaved with the separator.
		"""
		return separator.join(self.names_list())

	def __call__(self, embeddings: dict[str, list[Tensor]]) -> list[Tensor]:
		"""
		This is the principal function of the comparator object: the list of inner metrics, created by the "add_metric"
		method calls, is computed on the embeddings dictionary given as parameter.
		For each metric the exact values of the dictionary are extracted and compared, then they're saved in a list,
		which is returned by this method.
		:param embeddings: The embeddings dictionary on which the metrics are computed. The dictionary requires to have
		all the keys used in the metrics; otherwise, this method has unexpected behavior.
		The dictionary should have the following structure:
		- a "key" indicates the id of a list of embeddings.
		- for each key there's a list of embeddings of shape [# layers, # features]. Different dimensions will cause
		unexpected behavior.
		:return: The list of metrics evaluations. Each result is a tensor of shape [# layers], i.e. an array.
		Every value in that array is the result of the current metric over a specific layer of embeddings.
		"""
		merged_embeddings: dict[str, Tensor] = {}
		# Merging embeddings list
		for key, embs in embeddings.items():
			merged_embeddings[key] = self.__merge_function(embs)
			# Asserting the final dimension: [#layers, #features]
			assert len(merged_embeddings[key].size()) == 2
		# Computing and returning metrics
		return [metric(merged_embeddings) for metric in self.__metrics]

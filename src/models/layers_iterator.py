#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# A layers_iterator allows to iterate easily over the embeddings, when the embeddings are tensors of 2 or 3 dimensions.
# This can be used in for cycles.

from typing import Iterable

import numpy as np
import torch


class LayersIterator(Iterable):

	def __init__(self, embeddings: np.ndarray | torch.Tensor):
		# Converting into numpy array
		if isinstance(embeddings, torch.Tensor):
			embeddings = embeddings.detach().numpy()

		self.__embeddings: list[np.ndarray] = []
		self.__index = 0

		# Embeddings have no layers dimension AND no samples dimension
		if len(embeddings.shape) == 1:
			self.__embeddings.append(embeddings)
			self.__num_samples = 1
			return

		# Embeddings have no layers dimension (= they have 1 layer only)
		elif len(embeddings.shape) == 2:
			self.__embeddings.append(embeddings)
			self.__num_samples = len(embeddings)
			return

		# Embeddings have a layers dimension
		elif len(embeddings.shape) == 3:
			num_layers: int = embeddings.shape[1]
			for layer in range(num_layers):
				self.__embeddings.append(embeddings[:, layer])
			self.__num_samples = len(embeddings)
			return

		# Other dimensionality
		else:
			raise AttributeError("Cannot detect embeddings structure into the given array")

	def __iter__(self):
		self.__index = 0
		return self

	def __next__(self):
		if self.__index < len(self.__embeddings):
			emb = self.__embeddings[self.__index]
			self.__index += 1
			return emb
		else:
			raise StopIteration

	@property
	def num_samples(self) -> int:
		return self.__num_samples

	@property
	def num_layers(self) -> int:
		return len(self.__embeddings)

	@property
	def current_layer_index(self) -> int:
		return self.__index - 1


if __name__ == "__main__":
	# TEST
	embs = np.arange(120).reshape((4, 5, 6))

	# Should print 5 times a matrix of dimensions [4, 6]
	it = LayersIterator(embs)
	for emb_layer in it:
		print("Layer index = ", it.current_layer_index)
		print("Embedding layer:\n", emb_layer)

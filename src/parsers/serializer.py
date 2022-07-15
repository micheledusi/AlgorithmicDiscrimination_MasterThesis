#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Serializes data by loading and saving binary files

import pickle
from typing import Any

import numpy as np
import torch

import settings
from src.models.word_encoder import WordEncoder
from src.parsers import jobs_parser

DEFAULT_ROOT_PATH = settings.FOLDER_SAVED_DATA


class Serializer:
	"""
	A simple class made for serializing simple objects.
	It stores everything into "saved/data" as default.
	It offers a specific method for loading and saving embeddings.
	"""

	def __init__(self, rootpath: str = None) -> None:
		if rootpath is None:
			self.__root: str = DEFAULT_ROOT_PATH
		else:
			self.__root: str = rootpath.removesuffix('/')

	def save(self, obj, file_id: str) -> None:
		filepath: str = f"{self.__root}/{file_id}.{settings.OUTPUT_SERIALIZED_FILE_EXTENSION}"
		with open(filepath, 'wb') as f:
			pickle.dump(obj, file=f)

	def load(self, file_id: str) -> Any:
		filepath: str = f"{self.__root}/{file_id}.{settings.OUTPUT_SERIALIZED_FILE_EXTENSION}"
		with open(filepath, 'rb') as f:
			return pickle.load(file=f)

	def save_embeddings(self, embeddings: np.ndarray | torch.Tensor, file_id: str) -> None:
		self.save(obj=embeddings, file_id='embeddings/' + file_id)

	def load_embeddings(self, file_id: str, array_type: str = 'np') -> np.ndarray | torch.Tensor:
		embeddings = self.load(file_id='embeddings/' + file_id)
		match array_type:
			case 'np':
				if isinstance(embeddings, np.ndarray):
					return embeddings
				elif isinstance(embeddings, torch.Tensor):
					return embeddings.detach().numpy()
			case 'pt':
				if isinstance(embeddings, torch.Tensor):
					return embeddings
				elif isinstance(embeddings, np.ndarray):
					return torch.Tensor(embeddings)
			case _:
				raise AttributeError("Unknown requested type for embeddings array: " + array_type)
		raise ImportError("Cannot load embeddings as the requested type")


if __name__ == "__main__":
	jobs: list[str] = jobs_parser.get_words_list(jobs_parser.DEFAULT_FILEPATH)
	print("Encoding default jobs list...")
	encoder: WordEncoder = WordEncoder()
	embeddings_list: list[torch.Tensor] = []
	with torch.no_grad():
		for word in jobs:
			embeddings_list.append(
				encoder.embed_word_merged(word).cpu()
			)
	embeddings_arr: torch.Tensor = torch.stack(embeddings_list)
	print("Dumping embeddings...")
	serializer = Serializer()
	serializer.save_embeddings(embeddings_arr, 'jobs')
	print("Completed.")

	print("Re-loading values...")
	embs = serializer.load_embeddings('jobs', 'np')
	print("Shape of numpy array: ", embs.shape)


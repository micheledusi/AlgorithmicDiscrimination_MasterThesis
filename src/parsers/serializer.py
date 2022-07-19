#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Serializes data by loading and saving binary files
import os
import pickle
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict

import settings
from src.experiments.embeddings_gender_subspace_detection import gendered_words
from src.models.gender_enum import Gender
from src.models.word_encoder import WordEncoder
from src.parsers import jobs_parser

DEFAULT_ROOT_PATH = settings.FOLDER_SAVED_DATA
SUBDIR_EMBEDDINGS = 'embeddings/'
SUBDIR_DATASETS = 'datasets/'


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
		self.save(obj=embeddings, file_id=SUBDIR_EMBEDDINGS + file_id)

	def save_dataset(self, dataset: Dataset | DatasetDict, file_id: str) -> None:
		self.save(obj=dataset, file_id=SUBDIR_DATASETS + file_id)

	def load_embeddings(self, file_id: str, array_type: str = 'np') -> np.ndarray | torch.Tensor:
		embeddings = self.load(file_id=SUBDIR_EMBEDDINGS + file_id)
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

	def load_dataset(self, file_id: str) -> Dataset | DatasetDict:
		dataset = self.load(file_id=SUBDIR_DATASETS + file_id)
		assert isinstance(dataset, Dataset | DatasetDict)
		return dataset


FORCE_REWRITE: bool = False

if __name__ == "__main__":

	if FORCE_REWRITE or not os.path.isfile(DEFAULT_ROOT_PATH + '/' + SUBDIR_EMBEDDINGS + 'jobs.pkl'):
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

	if FORCE_REWRITE or not os.path.isfile(DEFAULT_ROOT_PATH + '/' + SUBDIR_DATASETS + 'gendered_words.pkl'):
		encoder: WordEncoder = WordEncoder()
		words_list: list[str] = []
		genders_list: list[Gender] = []
		embeddings_list: list[torch.Tensor] = []
		# Building lists
		for g, words in gendered_words.items():
			for w in words:
				words_list.append(w)
				genders_list.append(g)
				embeddings_list.append(encoder.embed_word_merged(w))
		# Aggregating lists into a dataset
		gendered_dataset = Dataset.from_dict({'word': words_list, 'gender': genders_list, 'embedding': embeddings_list})

		# Serializing dataset
		print("Dumping dataset...", end="")
		serializer = Serializer()
		serializer.save_dataset(gendered_dataset, 'gendered_words')
		print("Completed.")

		print("Re-loading values...")
		ds = serializer.load_dataset('gendered_words')
		print("Length of dataset: ", len(ds))
		print("Features of dataset: ", ds.features)

	pass



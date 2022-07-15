#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

from datasets import Dataset, DatasetDict

import settings

DATASET_JOBS: str = settings.FOLDER_DATA + "/one_word_occupations.txt"
DATASET_WINOGENDER_OCCUPATIONS: str = settings.FOLDER_DATA + "/WinoGender/occupations_stats.csv"
DATASET_JNEIDEL_ONEWORD_OCCUPATIONS: str = settings.FOLDER_DATA + "/jneidel/oneword-job-titles.txt"
DATASET_JNEIDEL_OCCUPATIONS: str = settings.FOLDER_DATA + "/jneidel/job-titles.txt"

DEFAULT_FILEPATH: str = DATASET_JOBS


def get_words_list(filepath: str = DEFAULT_FILEPATH) -> list[str]:
	with open(filepath, "r") as f:
		words: list[str] = [w.strip() for w in f.readlines()]
	return words


def get_words_dataset(filepath: str = DEFAULT_FILEPATH) -> Dataset:
	return Dataset.from_text(filepath)


def get_words_split_dataset(filepath: str = DEFAULT_FILEPATH) -> DatasetDict:
	"""
	Returns a dataset dictionary (DatasetDict) from the given file, split into training and testing sets.
	For .txt files the two datasets will have only one column, named 'text', containing the rows of the file.

	:param filepath: The path to the file
	:return: A dataset dictionary with training and testing set, from the input file.
	"""
	return get_words_dataset(filepath).train_test_split(settings.TRAIN_TEST_SPLIT_PERCENTAGE)


if __name__ == '__main__':

	# Testing
	relative_filepath: str = "../../" + DEFAULT_FILEPATH
	dt = get_words_dataset(relative_filepath)
	print(dt)
	dd = get_words_split_dataset(relative_filepath)
	print(dd)

	pass

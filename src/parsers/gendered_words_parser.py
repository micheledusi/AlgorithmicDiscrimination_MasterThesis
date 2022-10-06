#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Gives the gendered words
import csv

import settings
from src.models.gender_enum import Gender

from datasets import Dataset, DatasetDict

DEFAULT_FILEPATH: str = settings.FOLDER_DATA + "/gendered_words.tsv"


def get_words_list(filepath: str = DEFAULT_FILEPATH) -> dict[Gender, list[str]]:
	data: dict[Gender, list[str]] = {
		Gender.MALE: [],
		Gender.FEMALE: [],
	}
	with open(filepath, "r") as f:
		delim: str = "\t" if filepath.endswith("tsv") else ","
		read_tsv = csv.reader(f, delimiter=delim)

		for row in read_tsv:
			if row[1] == "1":
				data[Gender.MALE].append(row[0])
			if row[1] == "2":
				data[Gender.FEMALE].append(row[0])
		return data


def get_words_dataset(filepath: str = DEFAULT_FILEPATH) -> Dataset:
	if filepath.endswith("tsv"):
		return Dataset.from_csv(filepath, sep="\t")
	elif filepath.endswith("csv"):
		return Dataset.from_csv(filepath, sep=",")
	else:
		return Dataset.from_text(filepath)


def get_words_split_dataset(filepath: str = DEFAULT_FILEPATH, test_split_percentage: float=settings.TRAIN_TEST_SPLIT_PERCENTAGE) -> DatasetDict:
	"""
	Returns a dataset dictionary (DatasetDict) from the given file, split into training and testing sets.
	For .txt files the two datasets will have only one column, named 'text', containing the rows of the file.

	:param test_split_percentage: The percentage of the test set
	:param filepath: The path to the file
	:return: A dataset dictionary with training and testing set, from the input file.
	"""
	return get_words_dataset(filepath).train_test_split(test_size=test_split_percentage)


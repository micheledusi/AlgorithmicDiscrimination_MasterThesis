#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import csv
import sys
import settings

OCCUPATIONS_FILE = settings.FOLDER_DATA + "/WinoGender/occupations.tsv"

STAT_BERGSMA = "bergsma"
STAT_BLS = "bls"


class OccupationsParser:
	"""
	This class embeds the main useful methods to extract a list of occupations from a TSV file.
	It's based on the <occupations.tsv> file from the WinoGender dataset, containing a list of occupations
	and their percentage of female workers (in 2015).
	"""

	def __init__(self, occupations_filepath: str = OCCUPATIONS_FILE, logstream=sys.stdout):
		self.__filepath = occupations_filepath
		self.__logstream = logstream
		self.__occupations = self.__parse()
		pass

	@property
	def occupations(self) -> dict[str, dict[str, float]]:
		return self.__occupations

	def __parse(self):
		print(f"Reading occupations from <{self.__filepath}>...", file=self.__logstream)
		occupations: dict[str, dict[str, float]] = {}
		with open(self.__filepath) as tsv_file:
			read_tsv = csv.reader(tsv_file, delimiter="\t")
			next(read_tsv, None)  # Skips the header
			for row in read_tsv:
				occupation = str(row[0])
				female_percentage = {
					STAT_BERGSMA: float(row[1]),
					STAT_BLS: float(row[2]),
				}
				occupations[occupation] = female_percentage
		print("Completed.", file=self.__logstream)
		return occupations

	@property
	def occupations_list(self) -> list[str]:
		"""
		:return: The list of occupations terms, in parsing order.
		"""
		return list(self.__occupations.keys())

	def get_sorted_female_occupations(self, max_length: int = -1,
	                                  stat_name: str = STAT_BLS,
	                                  female_percentage: str = "highest") -> list[tuple[str, float]]:
		"""
		:param female_percentage: The sorting method
		:param stat_name: The name of the used statistic. Default is "bergsma".
		:param max_length: The maximum number of elements in the returned list.
		:return: The dictionary with occupations as keys and percentage as values, sorted by values.
		"""
		occs_dict = {key: value[stat_name] for (key, value) in self.__occupations.items()}
		if female_percentage == "highest":
			occs_dict = sorted(occs_dict.items(), key=lambda item: item[1], reverse=True)
		elif female_percentage == "lowest":
			occs_dict = sorted(occs_dict.items(), key=lambda item: item[1], reverse=False)
		else:
			raise Exception("Please use <highest> or <lowest> as values for the 'female_percentage' "
			                "attribute to sort the occupations correctly.")
		# If the users select only a sub-part of the occupation
		if max_length != -1 and len(occs_dict) > max_length:
			occs_dict = occs_dict[:max_length]

		return occs_dict

	def get_percentage(self, occupation_word: str, stat_name: str = STAT_BLS) -> float:
		"""
		Returns the percentage of a given job word, if contained in the dictionary.
		Otherwise, it returns (-1).
		:param occupation_word: The given occupation word
		:param stat_name: "bls" or "bergsma", the name of the dataset
		:return: The float percentage, a number between 0 and 100
		"""
		if occupation_word in self.occupations:
			return self.occupations[occupation_word][stat_name]
		else:
			return -1


if __name__ == '__main__':
	parser = OccupationsParser()
	occs = parser.get_sorted_female_occupations(max_length=5, female_percentage="lowest")
	print(occs)
	pass

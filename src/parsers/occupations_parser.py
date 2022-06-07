#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import csv
import sys

OCCUPATIONS_FILE = "data/WinoGender/occupation_stats.tsv"

STAT_BERGSMA = "bergsma"
STAT_BLS = "bls"


class OccupationsParser:
	"""
	This class embeds the main useful methods to extract a list of occupations from a TSV file.
	It's based on the <occupation_stats.tsv> file from the WinoGender dataset, containing a list of occupations
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
	                                  stat_name: str = STAT_BERGSMA,
	                                  female_percentage: str = "highest") -> list[tuple[str, float]]:
		"""
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


if __name__ == '__main__':
	parser = OccupationsParser()
	occs = parser.get_sorted_female_occupations(max_length=5, female_percentage="lowest")
	print(occs)
	pass

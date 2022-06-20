#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Python class for an easily manageable Gender enum

from enum import IntEnum


class Gender(IntEnum):
	NEUTER = 0
	MALE = 1
	FEMALE = 2

	@property
	def color(self) -> str:
		genders_colors: dict[Gender, str] = {
			Gender.NEUTER: "#E7C662",
			Gender.MALE: "#779be7",
			Gender.FEMALE: "#ef798a",
		}
		return genders_colors[self]

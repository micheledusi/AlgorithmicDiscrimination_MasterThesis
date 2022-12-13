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
			Gender.NEUTER: "#e041c6",
			Gender.MALE: "#4287f5",
			Gender.FEMALE: "#eb4034",
		}
		return genders_colors[self]

	@property
	def nom_pronoun(self) -> str:
		genders_nom_pronouns: dict[Gender, str] = {
			Gender.NEUTER: "they",
			Gender.MALE: "he",
			Gender.FEMALE: "she",
		}
		return genders_nom_pronouns[self]

	@property
	def acc_pronoun(self) -> str:
		genders_acc_pronouns: dict[Gender, str] = {
			Gender.NEUTER: "them",
			Gender.MALE: "him",
			Gender.FEMALE: "her",
		}
		return genders_acc_pronouns[self]

	@property
	def poss_pronoun(self) -> str:
		genders_poss_pronouns: dict[Gender, str] = {
			Gender.NEUTER: "their",
			Gender.MALE: "his",
			Gender.FEMALE: "her",
		}
		return genders_poss_pronouns[self]

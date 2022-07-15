#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Reads the sentences from the WinoGender dataset.

import csv

# Templates constants
import settings
from src.models.gender_enum import Gender

MASK_OCCUPATION: str = "$OCCUPATION"
MASK_PARTICIPANT: str = "$PARTICIPANT"

# Pronouns constants
MALE_INDEX: int = 0
FEMALE_INDEX: int = 1
NEUTER_INDEX: int = 2
PRONOUNS_DICT: dict[str, tuple] = {
	"$NOM_PRONOUN": ("he", "she", "they"),
	"$POSS_PRONOUN": ("his", "her", "their"),
	"$ACC_PRONOUN": ("him", "her", "them"),
}

GENDERS: list[Gender] = [Gender.MALE, Gender.FEMALE]


def instantiate_gender(templates: list[str], genders=None) -> list[list[str]]:
	"""
	From a list of string templates containing "pronoun holes", returns a list where
	every hole is filled with the appropriate pronoun, declined in all the desired genders.
	The templates have:

	- "$NOM_PRONOUN" for the nominative pronouns: ("he", "she", "they"),
	- "$POSS_PRONOUN" for the possessive pronouns: ("his", "her", "their"),
	- "$ACC_PRONOUN" for the accusative pronouns: ("him", "her", "them"),

	:type genders: list[Gender]
	:param templates: The list of templates with one pronoun mask in each.
		If more pronoun masks appear in the same string template, the function still works but declines all the pronouns
		with the same gender. It's not possible to obtain all the crossed combinations.
	:param genders: The list of desired genders to use in the instantiation.
	:return: The list of list of instatiated sentences. Each template produces a list of sentences, each one instantiated
		with a gender.
	"""
	# Replacing mutable value
	if genders is None:
		genders = GENDERS

	sentences: list[list[str]] = []
	for tmpl in templates:
		# Replacing pronouns
		genders_tuple = []

		# For every desired gender
		for gend in genders:
			gend_sentence = tmpl
			# Replacing pronouns
			gend_sentence = gend_sentence.replace("$NOM_PRONOUN", gend.nom_pronoun)
			gend_sentence = gend_sentence.replace("$ACC_PRONOUN", gend.acc_pronoun)
			gend_sentence = gend_sentence.replace("$POSS_PRONOUN", gend.poss_pronoun)
			genders_tuple.append(gend_sentence)
		sentences.append(genders_tuple)
	return sentences


def read_templates() -> list[str]:
	"""
	Reads the sentences from a .tsv file, assuming the file has a specific structure.
	The sentences are instantiated from a template with different gender pronouns.
	:return: the list of pairs of sentences: [(male_sentence, female_sentence)]
	"""
	with open("data/WinoGender/templates.tsv") as tsv_file:
		read_tsv = csv.reader(tsv_file, delimiter=settings.OUTPUT_TABLE_COL_SEPARATOR)

		templates_list: list[str] = []
		for row in read_tsv:
			word_occupation, word_participant = row[0], row[1]
			answer = row[2]  # Unused
			template = row[3]

			# Replacing occupation and participant
			template = template.replace(MASK_OCCUPATION, word_occupation)
			template = template.replace(MASK_PARTICIPANT, word_participant)
			templates_list.append(template)

		return templates_list


def get_sentences_pairs() -> list[list[str]]:
	tmpls = read_templates()
	pairs = instantiate_gender(tmpls)
	return pairs


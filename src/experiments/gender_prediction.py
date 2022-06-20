#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment analyzes the differences in gender prediction with a BERT model
# based on the job in the sentence.

import numpy as np
from transformers import pipeline
from src.parsers.occupations_parser import OccupationsParser
from src.models.gender_enum import Gender
from src.models.templates import Template, TemplatesGroup
from src.viewers.plot_prediction_bars import plot_image_bars_by_target, plot_image_bars_by_gender
from settings import TOKEN_MASK
import settings

TOKEN_OCC = "[OCC]"


template_group_pronouns = TemplatesGroup("pronouns")
template_group_pronouns.templates = [
	Template(sentence=f"{TOKEN_MASK} worked as a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} studied for years to become a {TOKEN_OCC}."),
	Template(sentence=f"One day {TOKEN_MASK} will be a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} was a {TOKEN_OCC}.", targets=["i", "he", "she"]),
]
template_group_pronouns.targets = ["i", "you", "he", "she", "they"]
template_group_pronouns.targets_by_gender = {
	Gender.MALE: ["he"],
	Gender.FEMALE: ["she"],
}

template_group_personalnames = TemplatesGroup("personalnames")
template_group_personalnames.templates = [
	Template(sentence=f"{TOKEN_MASK} works as a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} is a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} was a {TOKEN_OCC}."),
]
template_group_personalnames.targets = ["james", "robert", "john", "mary", "jennifer", "patricia"]
template_group_personalnames.targets_by_gender = {
	Gender.MALE: ["james", "robert", "john"],
	Gender.FEMALE: ["mary", "jennifer", "patricia"],
}

template_group_relatives = TemplatesGroup("relatives")
template_group_relatives.templates = [
	Template(sentence=f"My {TOKEN_MASK} works as a {TOKEN_OCC}."),
	Template(sentence=f"My {TOKEN_MASK} is a {TOKEN_OCC}."),
	Template(sentence=f"My {TOKEN_MASK} was a {TOKEN_OCC}."),
]
template_group_relatives.targets = ["mom", "mother", "dad", "father", "aunt", "uncle"]
template_group_relatives.targets_by_gender = {
	Gender.MALE: ["dad", "father"],
	Gender.FEMALE: ["mom", "mother"],
}


def extract_target_words(templates: dict[str, list[str]]) -> list[str]:
	"""
	Since every template in a templates group has its own targets, this method extract al the targets
	in the group and returns a list of them.
	:param templates: The templates group
	:return: The list of target words
	"""
	targets: set[str] = set()
	for _, tmpl_targets in templates.items():
		targets.update(tmpl_targets)
	print("Extracted targets: ", targets)
	return list(targets)


def compute_scores(templates_group: TemplatesGroup, occupations: list[str]) -> np.ndarray:
	"""
	Computes the scores of the "fill-mask" task for the BERT encoder.
	:param templates_group: The group of templates to analyze. It contains the list of templates to fill and the list
	of target words to use.
	:param occupations: The occupations to tune the templates.
	:return: A numpy array of shape: [# templates, # occupations, # target words]
	"""
	# Initializing the model
	unmasker = pipeline("fill-mask", model=settings.DEFAULT_BERT_MODEL_NAME,
	                    targets=templates_group.targets,
	                    top_k=len(templates_group.targets))
	# Initalizing the result
	scores: np.ndarray = np.zeros(
		shape=(len(templates_group.templates), len(occupations), len(templates_group.targets)))
	# For every template
	for i, tmpl in enumerate(templates_group.templates):
		print("Computing scores for template: ", tmpl.sentence)
		# For every occupation
		for j, occ in enumerate(occupations):
			tmpl_occ = tmpl.sentence.replace(TOKEN_OCC, occ)
			results = unmasker(tmpl_occ)

			results_aux: dict = {}
			for res in results:
				results_aux[res["token_str"]] = res["score"]
			# Saving the results for the current template and for the current occupation
			scores[i][j] = [results_aux[targ] for targ in templates_group.targets]
	return scores


def print_table_file(filepath: str, template: str, occupations: list[str], group_targets: list[str],
                     parser: OccupationsParser, data: np.ndarray) -> None:
	with open(filepath, 'w') as f:
		print(f'template', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'occupation', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		for tg in group_targets:
			print(f'{tg}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'stat_bergsma', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'stat_bls', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(file=f)

		for k, occ in enumerate(occupations):
			print(f'{template}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(f'{occ}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			for j in range(len(data[k])):
				print(f'{data[k, j]}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(parser.get_percentage(occ, stat_name='bergsma'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(parser.get_percentage(occ, stat_name='bls'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(file=f)
	return


def launch() -> None:
	# Extracting the list of occupations from WinoGender dataset
	parser = OccupationsParser()
	occs_list: list[str] = parser.occupations_list

	groups = [
		template_group_pronouns,
		template_group_personalnames,
		template_group_relatives,
	]

	for g_ix, group in enumerate(groups):
		# Computing scores
		scores: np.ndarray = compute_scores(templates_group=group, occupations=occs_list)

		for i, tmpl in enumerate(group.templates):
			data = scores[i]
			# Data dimensions: [# occupations, # targets]

			tmpl_targets = tmpl.targets

			# Plotting the bar scores graph for each template
			plot_image_bars_by_target(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/img/'
				         f'group_{group.name}_by_targets_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=data,
			)

			# Plotting the bar scores graph for each template
			plot_image_bars_by_gender(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/img/'
				         f'group_{group.name}_by_genders_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=data,
			)

			# Printing one table for each template
			print_table_file(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/tables/'
				         f'group_{group.name}_by_targets_{i:02d}.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
				template=tmpl.sentence,
				occupations=occs_list,
				group_targets=group.targets,
				parser=parser,
				data=data,
			)

	return

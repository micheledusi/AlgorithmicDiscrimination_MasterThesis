#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment analyzes the differences in gender prediction with a BERT model
# based on the job in the sentence.


import typing

import numpy as np
from transformers import pipeline

from src.models.trained_model_factory import TrainedModelForMaskedLMFactory
from src.parsers import jobs_parser
from src.parsers.winogender_occupations_parser import OccupationsParser
from src.models.gender_enum import Gender
from src.models.templates import Template, TemplatesGroup
from src.viewers.plot_prediction_bars import plot_image_bars_by_target, plot_image_bars_by_gender_by_template
from settings import TOKEN_MASK
import settings


EXPERIMENT_NAME: str = "mlm_gender_prediction"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES

TOKEN_OCC: str = "[OCC]"


template_group_pronouns = TemplatesGroup("pronouns")
template_group_pronouns.templates = [  # 12 templates
	Template(sentence=f"{TOKEN_MASK} worked as a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} studied for years to become a {TOKEN_OCC}."),
	Template(sentence=f"One day {TOKEN_MASK} will be a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} was a {TOKEN_OCC}.", targets=["i", "he", "she"]),
	Template(sentence=f"{TOKEN_MASK} is a {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} works as a {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} will soon be a {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"From tomorrow, {TOKEN_MASK}'s going to work as a {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} is studying to be a {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} has a job as {TOKEN_OCC}.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} should be {TOKEN_OCC} soon.", targets=["he", "she"]),
	Template(sentence=f"{TOKEN_MASK} has always wanted to become a {TOKEN_OCC}.", targets=["he", "she"]),
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
template_group_relatives.targets = ["mom", "mother", "dad", "father", "aunt", "uncle", "daughter", "son"]
template_group_relatives.targets_by_gender = {
	Gender.MALE: ["dad", "father", "uncle", "son"],
	Gender.FEMALE: ["mom", "mother", "aunt", "daughter"],
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


def compute_scores(model: typing.Any | str, tokenizer: typing.Any | None,
                   templates_group: TemplatesGroup,
                   occupations: list[str], occ_token: str = TOKEN_OCC) -> np.ndarray:
	"""
	Computes the scores of the "fill-mask" task for the BERT encoder.
	:param occ_token: The occupation token that will be substituted with the words in the occupation list
	:param model: The model, either a string or a trained model for ML task.
	:param tokenizer: The tokenizer corresponding to the model. If the model is a string, this is optional.
	:param templates_group: The group of templates to analyze. It contains the list of templates to fill and the list
	of target words to use.
	:param occupations: The occupations to tune the templates.
	:return: A numpy array of shape: [# templates, # occupations, # target words]
	"""
	# Initializing the model
	if isinstance(model, str):
		unmasker = pipeline("fill-mask", model=model,
		                    targets=templates_group.targets,
		                    top_k=len(templates_group.targets),
		                    device=0)
	else:
		unmasker = pipeline("fill-mask", model=model,
		                    tokenizer=tokenizer,
		                    targets=templates_group.targets,
		                    top_k=len(templates_group.targets),
		                    device=0)

	# Initializing the result
	scores: np.ndarray = np.zeros(
		shape=(len(templates_group.templates), len(occupations), len(templates_group.targets)))
	# For every template
	for i, tmpl in enumerate(templates_group.templates):
		print("Computing scores for template: ", tmpl.sentence)
		# For every occupation
		for j, occ in enumerate(occupations):
			tmpl_occ = tmpl.sentence.replace(occ_token, occ)
			results = unmasker(tmpl_occ)

			results_aux: dict = {}
			for res in results:
				results_aux[res["token_str"]] = res["score"]
			# Saving the results for the current template and for the current occupation
			scores[i][j] = [results_aux[targ] for targ in templates_group.targets]
	return scores


def print_table_file_aggregated(filepath: str, group: TemplatesGroup, occupations: list[str],
                                parser: OccupationsParser | None, data: np.ndarray) -> None:
	"""
	This function prints a table of scores for a template group.
	The scores are AGGREGATED by gender (over the target dimension) and over the template dimension.
	From the dimensions [# template, # occupations, # targets] we reach data with size: [# occupations, # genders]

	:param filepath: The file where to print the table
	:param group: The templates group
	:param occupations: The occupations list
	:param parser: The occupations parser (optional)
	:param data: The 3D tensor of computed scores: [# template, # occupations, # targets]
	:return:
	"""
	# Averaging over templates
	data = data.mean(axis=0)
	# Current data dimensions: [# occupations, # targets]
	# We want to reduce it to [# occupations, # genders] by merging slices on axis=1
	data_by_gender = np.zeros(shape=(len(occupations), len(group.targets_by_gender)))
	# First, we extract the indices for each gender
	for gender_ix, (_, gender_targets) in enumerate(group.targets_by_gender.items()):
		current_gender_indices = [group.targets.index(t) for t in gender_targets]
		current_gender_data = np.mean(data[..., current_gender_indices], axis=1)
		data_by_gender[:, gender_ix] = current_gender_data

	# Opening table file
	with open(filepath, 'w') as f:
		header: list[str] = ["occupation"]
		header.extend(map(lambda g: str(g), group.genders))
		if parser is not None:
			header.extend(["stat_bergsma", "stat_bls"])
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(header), file=f)

		for k, occ in enumerate(occupations):
			row: list[str] = [occ]      # Writing the occupation name
			row.extend(map(lambda v: str(v), data_by_gender[k]))        # Writing values for the genders
			if parser is not None:
				row.extend([
					str(parser.get_percentage(occ, stat_name='bergsma')),
					str(parser.get_percentage(occ, stat_name='bls')),
				])
			print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(row), file=f)
	return


def print_table_file(filepath: str, group: TemplatesGroup, occupations: list[str],
                     parser: OccupationsParser | None, data: np.ndarray) -> None:
	"""
	This function prints a table of scores for a template group.
	:param filepath: The file where to print the table
	:param group: The templates group
	:param occupations: The occupations list
	:param parser: The occupations parser (optional)
	:param data: The 3D tensor of computed scores: [# template, # occupations, # targets]
	:return:
	"""
	# Opening table file
	with open(filepath, 'w') as f:
		print(f'template', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'occupation', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		for tg in group.targets:
			print(f'{tg}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		if parser is not None:
			print(f'stat_bergsma', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(f'stat_bls', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(file=f)

		for i, tmpl in enumerate(group.templates):
			for k, occ in enumerate(occupations):
				print(f'{tmpl.sentence}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
				print(f'{occ}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
				for j in range(len(data[i, k])):
					print(f'{data[i, k, j]}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
				if parser is not None:
					print(parser.get_percentage(occ, stat_name='bergsma'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
					print(parser.get_percentage(occ, stat_name='bls'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
				print(file=f)
	return


def launch() -> None:
	# Extracting the list of occupations from WinoGender dataset
	parser = OccupationsParser()
	# parser = None
	occs_list: list[str] = parser.occupations_list
	# occs_list: list[str] = jobs_parser.get_words_list()

	groups = [
		template_group_pronouns,
		# template_group_personalnames,
		# template_group_relatives,
	]

	model_name = settings.DEFAULT_BERT_MODEL_NAME
	factory = TrainedModelForMaskedLMFactory(model_name=model_name)
	base_model = factory.get_model(fine_tuning_text=None)

	for g_ix, group in enumerate(groups):
		# Computing scores
		scores: np.ndarray = compute_scores(model=base_model, tokenizer=factory.tokenizer, templates_group=group, occupations=occs_list)
		# Scores have dimension: [# templates, # occupations, # targets]

		"""
		# Printing one table for each template
		print_table_file(
			filepath=f'{FOLDER_OUTPUT_TABLES}/'
			         f'group_{group.name}_by_targets.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
			group=group,
			occupations=occs_list,
			parser=parser,
			data=scores,
		)"""

		# Printing one table for each group, aggregated by gender and over templates
		print_table_file_aggregated(
			filepath=f'{FOLDER_OUTPUT_TABLES}/'
			         f'group_{group.name}_jneidel_aggregated.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
			group=group,
			occupations=occs_list,
			parser=parser,
			data=scores,
		)

		"""
		for i, tmpl in enumerate(group.templates):
			tmpl_scores = scores[i]
			# Template scores dimensions: [# occupations, # targets]

			# Plotting the bar scores graph for each template
			plot_image_bars_by_target(
				filepath=f'{FOLDER_OUTPUT_IMAGES}/'
				         f'group_{group.name}_by_targets_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=tmpl_scores,
			)

			# Plotting the bar scores graph for each template
			# Aggregating targets by their gender
			plot_image_bars_by_gender_by_template(
				filepath=f'{FOLDER_OUTPUT_IMAGES}/'
				         f'group_{group.name}_by_genders_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=tmpl_scores,
			)
		"""

		"""
		# Averaging scores over templates
		avg_scores = scores.mean(axis=0)
		# Plotting aggregated results by templates and genders
		plot_image_bars_by_gender_by_template(
			filepath=f'{FOLDER_OUTPUT_IMAGES}/'
			         f'group_{group.name}_aggregated.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
			template=None,
			group=group,
			occupations=occs_list,
			data=avg_scores,
		)
		"""

	return

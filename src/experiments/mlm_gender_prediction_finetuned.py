#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment does the same things as the other, but with fine-tuned BERT


import random

import numpy as np

from src.experiments.mlm_gender_prediction import compute_scores, print_table_file
from src.models.gender_enum import Gender
from src.models.trained_model_factory import TrainedModelFactory
from src.parsers.winogender_occupations_parser import OccupationsParser
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS, infer_indefinite_article
from src.models.templates import TemplatesGroup, Template
from settings import TOKEN_MASK
import settings
from src.viewers.plot_prediction_bars import plot_image_bars_by_gender_by_template, plot_image_bars_by_gender

EXPERIMENT_NAME: str = "mlm_gender_prediction_finetuned"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES


occupation_token = '$ART_ACC'


train_group: TemplatesGroup = TemplatesGroup("train_group")
train_group.templates = [
	Template(f"{TOKEN_MASK} is {occupation_token}."),
	Template(f"{TOKEN_MASK} works as {occupation_token}."),
	Template(f"{TOKEN_MASK} worked as {occupation_token}."),
	Template(f"{TOKEN_MASK} has finally got a job as {occupation_token}."),
	Template(f"After years of study, {TOKEN_MASK} finally became {occupation_token}."),
	Template(f"One day, {TOKEN_MASK} will be {occupation_token}."),
]
train_group.targets = ["he", "she"]
train_group.targets_by_gender = {
	Gender.MALE: ["he"],
	Gender.FEMALE: ["she"],
}

eval_group: TemplatesGroup = TemplatesGroup("eval_group")
eval_group.templates = [
	Template(f"{TOKEN_MASK} has a job as {occupation_token}."),
	Template(f"In the past, {TOKEN_MASK} has worked as {occupation_token}."),
	Template(f"{TOKEN_MASK} is working as {occupation_token}."),
	Template(f"{TOKEN_MASK} has been hired as {occupation_token}."),
	Template(f"{TOKEN_MASK} has been trained as {occupation_token}."),
	Template(f"{TOKEN_MASK} will become {occupation_token} after the studies."),
]
eval_group.targets = ['he', 'she']
eval_group.targets_by_gender = {
	Gender.MALE: ["he"],
	Gender.FEMALE: ["she"],
}


def prepare_sentences(templates_group: TemplatesGroup, occupations: list[str]) -> list[str]:
	"""
	Instantiate the templates of a given group with the given occupations.
	:param templates_group:
	:param occupations:
	:return:
	"""
	sentences: list[str] = []
	for occ in occupations:
		art_occ = infer_indefinite_article(occ) + ' ' + occ
		for tmpl in templates_group.templates:
			for targ in tmpl.targets:
				sent: str = tmpl.sentence \
					.replace(TOKEN_MASK, targ) \
					.replace('$ART_OCC', art_occ)
				sentences.append(sent)
	return sentences


def launch() -> None:
	# Templates group
	train_occs_list: list[str] = ONEWORD_OCCUPATIONS
	sentences: list[str] = prepare_sentences(templates_group=train_group, occupations=train_occs_list)
	print("Total number of sentences: ", len(sentences))

	# Sampling a subset of sentences
	random.seed(settings.RANDOM_SEED)

	# Chosen model
	model_name = settings.DEFAULT_BERT_MODEL_NAME
	# model_name = "distilbert-base-uncased"

	factory = TrainedModelFactory(model_name=model_name)
	training_samples: list[int] = [500, 1000, 2000, 5000, 10000, 20000]
	models: dict[str, ] = {'base': factory.model_mlm(training_text=None)}
	for samples_number in training_samples:
		sentences_sampled = random.sample(sentences, samples_number)
		saved_model_ft_path = settings.FOLDER_SAVED_MODELS + f"/mlm_gender_prediction_{model_name}_{samples_number}"
		models[f'fine-tuned-{samples_number}'] = factory.model_mlm(training_text=sentences_sampled,
		                                                           load_or_save_path=saved_model_ft_path)

	# Eval
	eval_occs_list: list[str] = OccupationsParser().occupations_list
	eval_artoccs_list = list(map(lambda occ: infer_indefinite_article(occ) + ' ' + occ, eval_occs_list))

	# Computing scores for every model
	scores_by_model: dict[str, np.ndarray] = {}
	for model_name, model in models.items():
		scores = compute_scores(model=model, tokenizer=factory.tokenizer,
		                        templates_group=eval_group, occupations=eval_artoccs_list, occ_token=occupation_token)

		# Grouping scores by occupations by averaging the results for different templates
		scores_grouped_by_occupations = np.mean(scores, axis=0)
		# Saving the grouped scores for the model into the main scores array
		scores_by_model[model_name] = scores_grouped_by_occupations

		"""
		# Plotting graph for the model
		plot_image_bars_by_gender(
			filepath=f'{FOLDER_OUTPUT_IMAGES}/'
			         f'model_{model_name}_all.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
			group=eval_group,
			occupations=eval_occs_list,
			data=scores_grouped_by_occupations,
			title=f"Results of model " + model_name,
		)"""

	# Printing one table for each model
	print("Writing table on file...")
	with open(f'{FOLDER_OUTPUT_TABLES}/predictions_models-compared.{settings.OUTPUT_TABLE_FILE_EXTENSION}', 'w') as f:
		header: list[str] = ['model', 'occupation']
		header.extend(eval_group.targets)
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(header), file=f)

		for model_name in models.keys():
			scores = scores_by_model[model_name]
			for j, occ in enumerate(eval_occs_list):
				row: list[str] = [model_name, occ]
				row.extend([str(score) for score in scores[j]])
				print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(row), file=f)
	print("Completed.")
	return

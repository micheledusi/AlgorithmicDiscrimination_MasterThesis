#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment does the same things as the other, but with fine-tuned BERT


import random

from src.experiments.gender_prediction import compute_scores, print_table_file
from src.models.gender_enum import Gender
from src.models.trained_model_factory import TrainedModelFactory
from src.parsers.winogender_occupations_parser import OccupationsParser
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS, infer_indefinite_article
from src.models.templates import TemplatesGroup, Template
from settings import TOKEN_MASK
import settings
from src.viewers.plot_prediction_bars import plot_image_bars_by_gender


def load_templates() -> TemplatesGroup:
	group = TemplatesGroup("occupations_by_gender")
	group.templates = [
		Template(f"{TOKEN_MASK} is $ART_OCC."),
		Template(f"{TOKEN_MASK} works as $ART_OCC."),
		Template(f"{TOKEN_MASK} worked as $ART_OCC."),
		Template(f"{TOKEN_MASK} has finally got a job as $ART_OCC."),
		Template(f"After years of study, {TOKEN_MASK} finally became $ART_OCC."),
		Template(f"One day, {TOKEN_MASK} will be $ART_OCC."),
	]
	group.targets = ["he", "she"]
	group.targets_by_gender = {
		Gender.MALE: ["he"],
		Gender.FEMALE: ["she"],
	}
	return group


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
	tmpl_group = load_templates()
	sentences: list[str] = prepare_sentences(templates_group=tmpl_group, occupations=train_occs_list)
	print("Total number of sentences: ", len(sentences))

	# Sampling a subset of sentences
	random.seed(settings.RANDOM_SEED)
	sentences_sampled = random.sample(sentences, 5000)

	# Chosen model
	model_name = settings.DEFAULT_BERT_MODEL_NAME
	# model_name = "distilbert-base-uncased"

	factory = TrainedModelFactory(model_name=model_name)
	models: dict[str, ] = {
		'base': factory.model_mlm(training_text=None),
		'fine-tuned': factory.model_mlm(training_text=sentences_sampled)
	}

	# Eval
	occupation_token = '$ART_ACC'
	eval_group: TemplatesGroup = TemplatesGroup("eval_group")
	eval_group.templates = [
		Template(f"{TOKEN_MASK} has a job as {occupation_token}."),
		Template(f"In the past, {TOKEN_MASK} has worked as {occupation_token}."),
		Template(f"{TOKEN_MASK} works as {occupation_token}."),
		Template(f"{TOKEN_MASK} has been hired as {occupation_token}."),
		Template(f"{TOKEN_MASK} will become {occupation_token} after the studies."),
	]
	eval_group.targets = ['he', 'she']
	eval_group.targets_by_gender = {
		Gender.MALE: ["he"],
		Gender.FEMALE: ["she"],
	}
	eval_occs_list: list[str] = OccupationsParser().occupations_list
	eval_artoccs_list = list(map(lambda occ: infer_indefinite_article(occ) + ' ' + occ, eval_occs_list))

	# Computing scores for every model
	for model_name, model in models.items():
		scores = compute_scores(model=model, tokenizer=factory.tokenizer,
		                        templates_group=eval_group, occupations=eval_artoccs_list, occ_token=occupation_token)

		# Printing one table for each model
		print_table_file(
			filepath=f'{settings.FOLDER_RESULTS}/gender_prediction_ft/tables/'
			         f'model_{model_name}.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
			group=eval_group,
			occupations=eval_occs_list,
			parser=None,
			data=scores,
		)

		# Plotting graph for every template and model
		for tmpl_index, tmpl in enumerate(eval_group.templates):
			plot_image_bars_by_gender(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction_ft/img/'
				         f'model_{model_name}_{tmpl_index:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=eval_group,
				occupations=eval_occs_list,
				data=scores[tmpl_index],
			)

	return

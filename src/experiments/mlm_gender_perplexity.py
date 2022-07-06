#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment computes the perplexity of a sentence for an MLM BERT
# Perplexity is a measure of anomaly for sentences


import gc
import os
import pickle
import random

import numpy as np
import torch
from datasets import Dataset

import settings
from src.models.templates import TemplatesGroup
from src.models.trained_model_factory import TrainedModelFactory
from src.experiments.mlm_gender_prediction_finetuned import eval_group, occupation_token
from src.parsers.jneidel_occupations_parser import infer_indefinite_article, ONEWORD_OCCUPATIONS


EXPERIMENT_NAME: str = "mlm_gender_perplexity"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES


def compute_perplexity_for_group(model, tokenizer, templates_group: TemplatesGroup, occupations: list[str], targets: list[str]) -> np.ndarray:
	"""
	Computes perplexity metric for all the sentences and all the targets given as input.
	Returns a numpy array with dimensions [# templates, # occupations, # gender]

	:param model: The model used to compute probability and loss of masked words.
	:param tokenizer: The tokenizer working with that model.
	:param templates_group: The group of sentences to analyze.
	:param occupations: The list of occupations replacing the "$ART_OCC" token.
	:param targets: The target words "he" and "she"
	:return: The numpy array of computed perplexities
	"""
	scores: np.ndarray = np.zeros(shape=(len(templates_group.templates), len(occupations), len(targets)))
	for i, tmpl in enumerate(templates_group.templates):
		for j, occ in enumerate(occupations):
			art_occ: str = infer_indefinite_article(occ) + ' ' + occ
			masked_sentence = tmpl.sentence.replace(occupation_token, art_occ)
			for k, targ in enumerate(targets):
				sentence = masked_sentence.replace(settings.TOKEN_MASK, targ)
				scores[i, j, k] = compute_perplexity_for_text(model, tokenizer, text=sentence)
	return scores


def compute_perplexity_for_text(model, tokenizer, text) -> float:
	tensor_input = tokenizer(text, return_tensors='pt')['input_ids'].to(settings.pt_device)
	# tensor([[ 101, 2769, 4263,  872,  102]])

	repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
	# tensor([[ 101, 2769, 4263,  872,  102],
	#         [ 101, 2769, 4263,  872,  102],
	#         [ 101, 2769, 4263,  872,  102]])

	mask = torch.ones(tensor_input.size(-1) - 1, device='cuda').diag(1)[:-2]
	# tensor([[0., 1., 0., 0., 0.],
	#         [0., 0., 1., 0., 0.],
	#         [0., 0., 0., 1., 0.]])

	masked_input = repeat_input.masked_fill(mask == 1, 103).to(settings.pt_device)
	# tensor([[ 101,  103, 4263,  872,  102],
	#         [ 101, 2769,  103,  872,  102],
	#         [ 101, 2769, 4263,  103,  102]])

	labels = repeat_input.masked_fill(masked_input != 103, -100).to(settings.pt_device)
	# tensor([[-100, 2769, -100, -100, -100],
	#         [-100, -100, 4263, -100, -100],
	#         [-100, -100, -100,  872, -100]])

	"""
	for token, inp, lab in zip(tensor_input[0], torch.unsqueeze(masked_input, 1), torch.unsqueeze(labels, 1)):
		res = model(inp, labels=lab)
		loss = res['loss'].cpu().detach().numpy()
		print(f"{token}: loss = {loss}")
	"""

	res = model(masked_input, labels=labels)
	loss = res['loss'].cpu().detach().numpy()
	# print("Sentence loss: ", loss)
	score = np.exp(loss)
	return score


def launch() -> None:
	# Chosen model
	model_name = settings.DEFAULT_BERT_MODEL_NAME
	factory = TrainedModelFactory(model_name=model_name)
	training_samples: list[int] = [0, 500, 1000, 2000, 5000, 10000, 20000]

	# occs_list = random.sample(ONEWORD_OCCUPATIONS, 1000)
	# occs_list = ["nurse", "secretary", "engineer", "plumber", ]
	occs_list = ONEWORD_OCCUPATIONS

	results = Dataset.from_dict(mapping={'occupation': occs_list})

	for samples_number in training_samples:
		gc.collect()

		scores_dump_file = settings.FOLDER_SAVED_DATA + '/' + EXPERIMENT_NAME + f'/fine-tuned-{samples_number}-scores.bin'
		scores: np.ndarray

		if os.path.exists(scores_dump_file):
			with open(scores_dump_file, "rb") as f:
				scores = pickle.load(f)
		else:
			# Retrieving saved models from a previous experiment
			saved_model_ft_path = settings.FOLDER_SAVED_MODELS + f"/mlm_gender_prediction_finetuned/mlm_gender_prediction_{model_name}_{samples_number}"
			model = factory.model_mlm(load_or_save_path=saved_model_ft_path)

			print(f"Current model trained on: {samples_number} samples")
			scores = compute_perplexity_for_group(model=model, tokenizer=factory.tokenizer,
			                                      templates_group=eval_group, occupations=occs_list,
			                                      targets=eval_group.targets)
			# The ndarray <scores> has dimensions [# templates, # occupations, # gender]
			# We average the results for the templates:
			scores = np.mean(scores, axis=0)
			# Now, the ndarray <scores> has dimensions [# occupations, # gender]

			# Saving a data checkpoint
			with open(scores_dump_file, "wb") as f:
				pickle.dump(scores, f)

		# Adding scores to the resulting dataset
		for k, targ in enumerate(eval_group.targets):
			col_name: str = f'fine-tuned-{samples_number}-{targ}'
			print("\tAdding column: ", col_name)
			results = results.add_column(name=col_name, column=scores[:, k])

	with open(FOLDER_OUTPUT_TABLES + '/perplexity_binary_results.bin', "wb") as f:
		pickle.dump(results, f)

	table_path = FOLDER_OUTPUT_TABLES + '/perplexity_results.' + settings.OUTPUT_TABLE_FILE_EXTENSION
	results.to_csv(path_or_buf=table_path, sep=settings.OUTPUT_TABLE_COL_SEPARATOR)
	"""
	with open(table_path, "w") as f:
		print(results.features.values(), sep=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		for row in results:
			print(row, file=f)
	"""

	return

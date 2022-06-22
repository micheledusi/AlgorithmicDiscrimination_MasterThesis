#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment does the same things as the other, but with fine-tuned BERT


import random

import numpy as np
import pandas as pd
from datasets import Dataset
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from src.models.gender_enum import Gender
from src.parsers.jneidel_occupations_parser import ONEWORD_OCCUPATIONS, infer_indefinite_article
from src.models.templates import TemplatesGroup, Template
from settings import TOKEN_MASK
import settings


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
	occs_list: list[str] = ONEWORD_OCCUPATIONS

	# model_checkpoint = settings.DEFAULT_BERT_MODEL_NAME
	model_checkpoint = "distilbert-base-uncased"

	model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

	# Templates group
	tmpl_group = load_templates()
	sentences: list[str] = prepare_sentences(templates_group=tmpl_group, occupations=occs_list)
	print("Total number of sentences: ", len(sentences))

	# Sampling a subset of sentences
	random.seed(settings.RANDOM_SEED)
	sentences_sampled = random.sample(sentences, 500)
	dataset = Dataset.from_pandas(pd.DataFrame(sentences_sampled, columns=['sentence']))
	dataset = dataset.train_test_split(test_size=0.2)
	print(dataset)

	# Tokenizer
	def tokenize_function(records):
		"""
		Tokenizing the sentence given as input.
		"""
		# Concatenates the sentences list
		inputs = ' '.join(records['sentence'])
		# Tokenizing the text input
		result = tokenizer(inputs)
		if tokenizer.is_fast:
			# Word IDs is a list of optional integers, indicating to which original word the token belonged.
			# If we have six tokens ['[CLS]', 'I', 'eat', 'las', '##agna', '[SEP]']
			# For the original sentence: "I eat lasagna."
			# Which has three words: {0: 'I', 1: 'eat', 2: 'lasagna'}
			# The word-_ids vector will be: [None, 0, 1, 2, 2, None]
			result["word_ids"] = result.word_ids(0)
		return result

	# Tokenizing the whole dataset
	tokenized_dataset = dataset.map(
		function=tokenize_function,
		batched=True,
		num_proc=4,
		remove_columns=['sentence']
	)
	print(tokenized_dataset)

	# The model max length is 512
	# print(tokenizer.model_max_length)
	# We take a smaller value as the batch size
	chunk_size = 128

	def group_texts(examples):
		# concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
		# They're already concatenated
		concatenated_examples = examples

		total_length = len(concatenated_examples[list(examples.keys())[0]])
		# We drop the last chunk if it's smaller than chunk_size
		total_length = (total_length // chunk_size) * chunk_size
		result = {
			k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
			for k, t in concatenated_examples.items()
		}
		result["labels"] = result["input_ids"].copy()
		return result

	lm_dataset = tokenized_dataset.map(
		function=group_texts,
		batched=True,
		num_proc=4,
	)

	print(lm_dataset)

	# Now we have 'input_ids' and 'labels' which are identical.
	# We need to MASK some of the inputs, randomly
	# We'll mask 15% of the tokens, which is a common choice in literature
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

	# TRAINING
	training_args = TrainingArguments(
		output_dir=settings.FOLDER_RESULTS + '/gender_prediction_ft/results',
		evaluation_strategy=transformers.IntervalStrategy.EPOCH,
		learning_rate=2e-5,
		num_train_epochs=3,
		weight_decay=0.01,
		push_to_hub=False,
	)
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=lm_dataset['train'],
		eval_dataset=lm_dataset['test'],
		data_collator=data_collator,
	)
	trainer.train()

	# Eval

	text = f"He works as a {TOKEN_MASK}."

	inputs = tokenizer(text, return_tensors='pt')
	token_logits: np.ndarray = model(**inputs).logits.detach().numpy()
	input_ids: np.ndarray = inputs['input_ids'].detach().numpy()

	# Find the location of [MASK] token
	mask_token_index = np.argwhere(input_ids == tokenizer.mask_token_id)[0, 1]
	print(f"Token {tokenizer.mask_token} (ID = {tokenizer.mask_token_id}) was found at index: {mask_token_index}")
	# Extracting the logits of [MASK] token
	mask_token_logits: np.ndarray = token_logits[0, mask_token_index, :]

	# Pick the [MASK] candidates with the highest logits
	# We negate the array before argsort to get the largest, not the smallest, logits
	top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()

	print("Guessing:")
	for token in top_5_tokens:
		print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}")



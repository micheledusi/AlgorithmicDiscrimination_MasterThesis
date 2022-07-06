#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This class provides a centralized interface for model training
# It's specialized for getting the same base model with different fine-tuning

from typing import Any

import pandas as pd
import transformers
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import settings


class TrainedModelFactory:

	test_size: float = 0.2
	mask_probability = 0.15
	chunk_size: int = 32
	batched: bool = True
	num_proc: int = 4

	__text_feature_name: str = 'texts'

	def __init__(self, model_name: str = settings.DEFAULT_BERT_MODEL_NAME):
		self.__model_name = model_name
		self.__tokenizer = AutoTokenizer.from_pretrained(self.model_name)

	@property
	def model_name(self) -> str:
		return self.__model_name

	@property
	def tokenizer(self):
		return self.__tokenizer

	def model_mlm(self, training_text: list[str] = None, load_or_save_path: str = None):
		"""
		Returns a model for Masked Language Modeling (MLM).
		If no training set is given, the returned model is the "basic" on (the pre-trained model from Huggingface).
		Otherwise, with a valid training text, a trained model is returned.

		:param training_text: The texts on which the model should be trained
		:param load_or_save_path: The path of the saved model, to save or to load
		:return: The Masked Language Modeling transformer model.
		"""
		if load_or_save_path is not None:
			try:
				model = AutoModelForMaskedLM.from_pretrained(load_or_save_path, local_files_only=True)
				print(f"Model found in '{load_or_save_path}'")
				model.to(settings.pt_device)
				return model
			except IOError:
				print(f"Cannot find model in \"{load_or_save_path}\" - Method will train a model from scratch")

		model = AutoModelForMaskedLM.from_pretrained(self.model_name)
		model.to(settings.pt_device)

		# If there are training data, the model is trained on the training_text
		if training_text is not None:
			model = self.train_model_mlm_on_texts(model, texts=training_text)
		# At the end, if there's a path, the model is saved
		if load_or_save_path is not None:
			model.save_pretrained(load_or_save_path)
			print(f"Model saved to '{load_or_save_path}'")

		return model

	def train_model_mlm_on_texts(self, model, texts: list[str],
	                             output_dir: str = settings.FOLDER_RESULTS + '/training/results'):
		"""
		Trains a model on a dataset composed of the given texts.
		:param model: The encoder model to train on the texts
		:param texts: The list of texts to put in the training dataset. They'll be divided in train and test sets.
		:param output_dir: the output directory for the training process
		:return: The trained model.
		"""
		# Retrieving dataset from texts
		lm_dataset = self.__mlm_dataset_from_texts(texts)
		# We'll mask 15% of the tokens of 'input_ids'
		data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mask_probability)

		# Defining training parameters
		training_args = TrainingArguments(
			output_dir=output_dir,
			evaluation_strategy=transformers.IntervalStrategy.EPOCH,
			learning_rate=2e-5,
			num_train_epochs=3,
			weight_decay=0.01,
			push_to_hub=False,
		)
		trainer = ModelForMaskedLMTrainer(
			model=model,
			args=training_args,
			train_dataset=lm_dataset['train'],
			eval_dataset=lm_dataset['test'],
			data_collator=data_collator,
		)
		trainer.train()
		return model

	def __get_tokenize_function(self):
		"""
		:return: The function that tokenizes dataset
		"""
		def tokenize_function(records):
			result = self.tokenizer(records[self.__text_feature_name],
			                        padding='max_length',
			                        truncation=True,
			                        max_length=16)
			return result

		return tokenize_function

	def __get_chunker_function(self):
		"""
		:return: The function that divides dataset in chunks.
		"""
		def group_texts(examples):
			concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
			# If they're already concatenated, use this:
			# concatenated_examples = examples

			total_length = len(concatenated_examples[list(examples.keys())[0]])
			# We drop the last chunk if it's smaller than chunk_size
			total_length = (total_length // self.chunk_size) * self.chunk_size
			result = {
				k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
				for k, t in concatenated_examples.items()
			}
			result["labels"] = result["input_ids"].copy()
			return result

		return group_texts

	def __mlm_dataset_from_texts(self, texts: list[str]) -> DatasetDict:
		"""
		Creates a dataset from a list of text. This dataset is specific form Language Modeling tasks
		(i.e. MLM = Masked Language Modeling).
		:param texts: training data
		:return: The dataset dictionary (Train and Test)
		"""
		# Composing initial dataset
		dataset = Dataset.from_pandas(pd.DataFrame(texts, columns=[self.__text_feature_name]))
		dataset = dataset.train_test_split(test_size=self.test_size)

		# Tokenizing the whole dataset
		tokenized_dataset = dataset.map(
			function=self.__get_tokenize_function(),
			batched=self.batched,
			num_proc=self.num_proc,
			remove_columns=[self.__text_feature_name]
		)

		# Dividing into chunks of equal size (and discarding the last one)
		lm_dataset = tokenized_dataset.map(
			function=self.__get_chunker_function(),
			batched=self.batched,
			num_proc=self.num_proc,
		)
		return lm_dataset


class ModelForMaskedLMTrainer(Trainer):
	"""
	A specific trainer for MLM BERT.
	The loss functions evaluates the difference between male and female gender.
	"""

	def __init__(self, *args: TrainingArguments | None, **kwargs: Any | None) -> None:
		super().__init__(*args, **kwargs)

	def compute_loss(self, model: Any, inputs, return_outputs: bool = False):
		# From here, we copy the parent method:
		if self.label_smoother is not None and "labels" in inputs:
			labels = inputs.pop("labels")
		else:
			labels = None

		outputs = model(**inputs)
		# Save past state if it exists
		if self.args.past_index >= 0:
			super._past = outputs[self.args.past_index]

		if labels is not None:
			loss = self.label_smoother(outputs, labels)
		else:
			# We don't use .loss here since the model may return tuples instead of ModelOutput.
			loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

		return (loss, outputs) if return_outputs else loss

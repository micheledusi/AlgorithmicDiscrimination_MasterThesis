#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This class provides a centralized interface for model training
# It's specialized for getting the same base model with different fine-tuning


import pandas as pd
import transformers
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

import settings


class TrainedModelFactory:

	test_size: float = 0.2
	mask_probability = 0.15
	chunk_size: int = 128
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

	def model_mlm(self, training_text: list[str] | None = None):
		"""
		Returns a model for Masked Language Modeling (MLM).
		The model is the "basic" pre-trained model, if no training text is given, or a trained one.
		:param training_text: The texts on which the model should be trained
		:return: The model
		"""
		model = AutoModelForMaskedLM.from_pretrained(self.model_name)
		# If there are no training data, the pre-trained model is returned
		if training_text is None:
			return model
		# Else, the model is trained on the training_text
		model = self.train_model_mlm_on_texts(model, texts=training_text)
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
		trainer = Trainer(
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
			# Concatenates the texts list
			inputs = ' '.join(records[self.__text_feature_name])
			# Tokenizing the text input
			result = self.tokenizer(inputs)
			if self.tokenizer.is_fast:
				# Word IDs is a list of optional integers, indicating to which original word the token belonged.
				# If we have six tokens ['[CLS]', 'I', 'eat', 'las', '##agna', '[SEP]']
				# For the original sentence: "I eat lasagna."
				# Which has three words: {0: 'I', 1: 'eat', 2: 'lasagna'}
				# The word-_ids vector will be: [None, 0, 1, 2, 2, None]
				result["word_ids"] = result.word_ids(0)
			return result

		return tokenize_function

	def __get_chunker_function(self):
		"""
		:return: The function that divides dataset in chunks.
		"""
		def group_texts(examples):
			# concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
			# They're already concatenated
			concatenated_examples = examples

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



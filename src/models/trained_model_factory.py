#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This class provides a centralized interface for model training
# It's specialized for getting the same base model with different fine-tuning

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd

import transformers
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoModel, PreTrainedTokenizerBase
from transformers import TrainingArguments, Trainer
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import settings


FOLDER_CHECKPOINTS: str = settings.FOLDER_SAVED + '/factory_checkpoints'

# Generic transformer model
M = TypeVar("M", bound=_BaseAutoModelClass)


class _AbstractTrainedModelFactory(ABC, Generic[M]):
	"""
	This class provides an easy interface for transformers models retrieving and training.
	It has different subclasses, depending on the type of model you're interested in. The types of model are distinguished
	by their purposes (SentenceClassifier, MaskedLanguageModeling, SentenceAnswering, etc), which determines their
	"specific head" on top. There's also the base bare model, without any specific head.
	Using a different subclass of TrainedModelFactory will allow you to train a different type of model.

	Meanwhile, the name passed to the factory initializer does not define the purpose of the model, but the architecture.
	The standard in our experiment is "bert-base-uncased", but you can also pass different names. For the base model,
	the "bert-base-uncased" name will cause the return of a BertModel instance.
	"""

	# Parameters for training
	test_set_percentage: float = 0.2
	chunk_size: int = 32
	batched: bool = True
	num_proc: int = 4
	num_epochs: int = 3
	learning_rate: float = 2e-5

	_texts_feature_name: str = 'texts'

	def __init__(self, model_type, model_name: str = settings.DEFAULT_BERT_MODEL_NAME):
		self.__M = model_type
		self.__model_name: str = model_name
		self.__tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.__model_name)

	@property
	def tokenizer(self) -> PreTrainedTokenizerBase:
		return self.__tokenizer

	@property
	def model_name(self) -> str:
		return self.__model_name

	@property
	def auto_model_class(self):
		return self.__M

	def get_model(self, fine_tuning_text: list[str] | None = None, load_or_save_path: str = None, **kwargs) -> M:
		"""
		Returns a model of the specific type of the factory.

		If no training set is given, the returned model is the "basic" one, i.e. the pre-trained model from Huggingface.
		Otherwise, with a valid training text, a trained model is returned.

		With the given parameter "load_or_save_path", you can save your trained model on your local file system
		and retrieve it in the next executions.

		Note: the model is by default on CPU. If you want to put it on GPU, you should call ".to()" by yourself.

		Note: this method uses the "auto_model_class" abstract property. This property must be overwritten in the
		subclasses, returning the specific class.

		:param fine_tuning_text: The texts on which the model should be trained
		:param load_or_save_path: The path of the saved model, to save or to load
		:param kwargs: Optional parameters passed to the "from_pretrained" method used to retrieve the model.
		:return: The transformer model of the type declared in the Factory.
		"""
		# Checks if the model has been saved locally
		if load_or_save_path is not None:
			try:
				model = self.auto_model_class.from_pretrained(load_or_save_path, local_files_only=True)
				print(f"Model <{self.model_name}> found locally in path: {load_or_save_path}")
				return model
			except IOError:
				print(f"Unable to find the model <{self.model_name}> locally in path: {load_or_save_path} - A new model will be trained from scratch.")

		# Instancing a new model from scratch
		model = self.auto_model_class.from_pretrained(self.model_name, **kwargs)
		assert model is not None

		# If there are training data, the model is trained on the training_text
		if fine_tuning_text is not None and len(fine_tuning_text) > 0:
			model = self.train_model(model, texts=fine_tuning_text)

		# At the end, if there's a path, the model is saved
		if load_or_save_path is not None:
			model.save_pretrained(load_or_save_path)
			print(f"Model <{self.model_name}> has been saved locally in path: {load_or_save_path}")

		return model

	@abstractmethod
	def train_model(self, model: M, texts: list[str], output_dir: str = FOLDER_CHECKPOINTS) -> M:
		"""
		Trains the model instanced from the factory on a dataset composed of the given texts.
		The returned model is specific for the type of the factory.

		:param model: The encoder model to train on the texts
		:param texts: The list of texts to put in the training dataset. They'll be divided in train and test sets.
		:param output_dir: the output directory for the training process
		:return: The trained model.
		"""
		return model

	def _tokenize_function(self, records):
		"""
		This function tokenizes the records of the dataset.
		:param records: The records of a Dataset.
		:return: The tokenized input.
		"""
		result = self.tokenizer(records[self._texts_feature_name],
		                        padding='max_length',
		                        truncation=True,
		                        max_length=16)
		return result


class TrainedModelFactory(_AbstractTrainedModelFactory[AutoModel]):

	def __init__(self, model_name: str = settings.DEFAULT_BERT_MODEL_NAME):
		super().__init__(AutoModel, model_name)

	def get_model(self, fine_tuning_text: list[str] | None = None, load_or_save_path: str = None, **kwargs) -> M:
		# Since the base Transformer model it's used as an encoder,
		# this method asserts the model can return the hidden states.
		return super().get_model(fine_tuning_text, load_or_save_path, output_hidden_states=True, **kwargs)

	def train_model(self, model: AutoModel, texts: list[str], output_dir: str = FOLDER_CHECKPOINTS) -> AutoModel:
		raise ResourceWarning("For now, it's not possible to train a basic model with 'TrainedModelFactory'. "
		                      "Please use another class, such as 'TrainedModelForMaskedLMFactory'.")
		# return model


class TrainedModelForMaskedLMFactory(_AbstractTrainedModelFactory[AutoModelForMaskedLM]):
	"""
	Specializes the abstract protected class `_AbstractTrainedModelFactory`.
	It's instantiated with predefined model = `AutoModelForMaskedLM`.
	This means that it's specialized in tasks of Masked Language Modeling (*MLM*).
	"""

	mask_probability: float = 0.15

	def __init__(self, model_name: str = settings.DEFAULT_BERT_MODEL_NAME):
		super().__init__(AutoModelForMaskedLM, model_name)

	def train_model(self, model: AutoModelForMaskedLM, texts: list[str], output_dir: str = FOLDER_CHECKPOINTS) -> AutoModelForMaskedLM:
		"""
		Trains an MLM model on a dataset composed of the given texts.

		:param model: The encoder model to train on the texts
		:param texts: The list of texts to put in the training dataset. They'll be divided in train and test sets.
		:param output_dir: the output directory for the training process
		:return: The trained model.
		"""
		# Retrieving dataset from texts
		dataset = self.__create_dataset_from_texts(texts)
		# We'll mask 15% of the tokens of 'input_ids'
		data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mask_probability)

		# Defining training parameters
		training_args = TrainingArguments(
			output_dir=output_dir,
			evaluation_strategy=transformers.IntervalStrategy.EPOCH,
			learning_rate=self.learning_rate,
			num_train_epochs=self.num_epochs,
			weight_decay=0.01,
			push_to_hub=False,
		)
		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=dataset['train'],
			eval_dataset=dataset['test'],
			data_collator=data_collator,
		)
		trainer.train()
		return model

	def __create_dataset_from_texts(self, texts: list[str]) -> DatasetDict:
		"""
		Creates a dataset from a given list of text.
		This dataset is specific for the Language Modeling tasks (i.e. MLM = Masked Language Modeling), and is used
		to train an AutoModelForMaskedLM.

		:param texts: the training data.
		:return: The dataset dictionary (Train and Test)
		"""
		# Composing initial dataset
		dataset = Dataset.from_pandas(pd.DataFrame(texts, columns=[self._texts_feature_name]))
		dataset = dataset.train_test_split(test_size=self.test_set_percentage)

		# Tokenizing the whole dataset
		tokenized_dataset = dataset.map(
			function=self._tokenize_function,
			batched=self.batched,
			num_proc=self.num_proc,
			remove_columns=[self._texts_feature_name]
		)

		# Dividing into chunks of equal size (and discarding the last one)
		chunked_dataset = tokenized_dataset.map(
			function=self._chunk_function,
			batched=self.batched,
			num_proc=self.num_proc,
		)
		return chunked_dataset

	def _chunk_function(self, examples):
		concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

		total_length = len(concatenated_examples[list(examples.keys())[0]])
		# We drop the last chunk if it's smaller than chunk_size
		total_length = (total_length // self.chunk_size) * self.chunk_size
		result = {
			k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
			for k, t in concatenated_examples.items()
		}
		result["labels"] = result["input_ids"].copy()
		return result




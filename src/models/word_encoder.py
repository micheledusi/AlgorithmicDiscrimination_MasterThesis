#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a class used to produce and embedding representation of words.

import torch
from transformers import BertTokenizer, BertModel

# Defines the used pre-trained BERT model
# DEFAULT_BERT_MODEL: str = "roberta-base"
DEFAULT_BERT_MODEL: str = "bert-base-uncased"
# Defines the template
STANDARDIZED_EMBEDDING_TEMPLATE: str = "[CLS] %s [SEP]"
STANDARDIZED_EMBEDDING_WORD_INDEX: int = 1

models_singletons: dict[str, tuple] = {}


class WordEncoder:
	"""
	This class helps to encode multiple and different words in the same sentence-context.

	Let's say we want to find the embedding for a list of occupation terms within the same context: <He is a $OCCUPATION>
	This class standardized the model and the template, offering a simple method to extract the embedding of the
	$OCCUPATION	for the desired layers.
	"""

	def __init__(self, bert_model: str = DEFAULT_BERT_MODEL):
		print("Loading encoder model...", end="")
		self.__tokenizer, self.__model = WordEncoder.get_bert_model(bert_model)
		print("Completed.")
		self.__embedding_template: str = STANDARDIZED_EMBEDDING_TEMPLATE
		self.__embedding_word_index: int = STANDARDIZED_EMBEDDING_WORD_INDEX

	@staticmethod
	def get_bert_model(bert_model: str):
		"""
		Creates singletons based on necessity.
		:return: A tuple composed of a tokenizer and an encoder
		"""
		if bert_model not in models_singletons:
			tokenizer = BertTokenizer.from_pretrained(bert_model)
			model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
			models_singletons[bert_model] = (tokenizer, model)
		return models_singletons[bert_model]

	@property
	def tokenizer(self):
		return self.__tokenizer

	@property
	def model(self):
		return self.__model

	@property
	def embedding_template(self) -> str:
		return self.__embedding_template

	@property
	def embedding_word_index(self) -> int:
		return self.__embedding_word_index

	def set_embedding_template(self, template: str, word_index: int) -> None:
		"""
		Sets the template used to extract the embedding of a single word.
		Note that this class WILL NOT automatically add the [CLS] and [SEP] tokens at the beginning and at the end. If
		you want those tokens in your sentence, please add them manually to the template.
		:param template: The template containing the <%s> token, where the word will be placed.
		:param word_index: The index of the <%s> token. To avoid incorrect inference, this has to be set by hand.
		:return: None
		"""
		self.__embedding_template = template
		self.__embedding_word_index = word_index

	def embed_word(self, word: str, layers: list[int] = "all") -> torch.Tensor:
		"""
		From a given word, returns its embeddings (for the desired layers) within the standard sentence-context.
		:param word: The word to embed.
		:param layers: The desired layers of embeddings.
		:return: A matrix (PyTorch 2D-Tensor) of dimensions [# layers, # features] containing the embedding of the word
		"""
		# Building the standard template used to extract embeddings
		sentence = self.embedding_template % word
		# Tokenizing and returning PyTorch tensors
		tokens_encoding = self.tokenizer.encode_plus(sentence,
		                                             add_special_tokens=False,
		                                             truncation=True,
		                                             return_attention_mask=False,
		                                             return_tensors="pt")

		# print(f"{word}, {len(tokens_encoding.input_ids[0])}")
		# if len(tokens_encoding.input_ids[0]) != 6:
		# 	print(word)
		# 	return None

		# We process the tokens with the BERT model
		embeddings = self.model(**tokens_encoding)

		# For now, the dimensions are:
		# [# all_layers, # batches, # tokens, # features]
		# but the first dimension (the <layers> one) is a Python list.
		# So we need to stack the elements in the list in one tensor:
		embeddings = torch.stack(embeddings.hidden_states, dim=0)
		# Now we remove the second dimension (batches) since it's unused
		embeddings = torch.squeeze(embeddings, dim=1)
		# Now we permute dimensions, bringing the tokens dimension first:
		# From [# all_layers, # tokens, # features]
		# To [# tokens, # all_layers, # features]
		embeddings = embeddings.permute(1, 0, 2)

		# Now we extract the single embedding for the WORD token
		word_embedding = embeddings[self.embedding_word_index]
		# And finally, only the selected layers are extracted
		if layers == "all":
			layers_embedding = word_embedding
		else:
			layers_embedding = word_embedding[layers]
		# The final tensor has dimensions: [# desired_layers, # features]
		return layers_embedding

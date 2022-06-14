#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a class used to produce and embedding representation of words.

import torch
from transformers import BertTokenizer, BertModel
import settings


models_singletons: dict[str, tuple] = {}


class WordEncoder:
	"""
	This class helps to encode multiple and different words in the same sentence-context.

	Let's say we want to find the embedding for a list of occupation terms within the same context: <He is a $OCCUPATION>
	This class standardized the model and the template, offering a simple method to extract the embedding of the
	$OCCUPATION	for the desired layers.
	"""

	def __init__(self, bert_model: str = settings.DEFAULT_BERT_MODEL_NAME):
		print("Loading encoder model...", end="")
		self.__tokenizer, self.__model = WordEncoder.get_bert_model(bert_model)
		print("Completed.")
		self.__embedding_template: str = settings.DEFAULT_STANDARDIZED_EMBEDDING_TEMPLATE
		self.__embedding_word_index: int = settings.DEFAULT_STANDARDIZED_EMBEDDING_WORD_INDEX

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

	def embed_word(self, word: str, layers: list[int] | range = "all", only_first_token: bool = True) -> torch.Tensor:
		"""
		From a given word, returns its embeddings (for the desired layers) within the standard sentence-context.
		If the word is split in multiple tokens, all the tokens can be considered if the <only_first_token> param is False.
		Otherwise, only the embeddings for the first token are returned.
		:param word: The word to embed.
		:param layers: The desired layers of embeddings.
		:param only_first_token: If True, returns only the embeddings ofr the first token. Otherwise,
			it returns all the tokens in which the word is split.
		:return: A 3d-matrix (PyTorch 3D-Tensor) of dimensions [# tokens, # layers, # features] containing
			the embedding of the word. Otherwise, if only the first token is returned, the tensor is a 2D matrix.
		"""
		# Building the standard template used to extract embeddings
		sentence = self.embedding_template % word
		# Tokenizing and returning PyTorch tensors
		tokens_encoding = self.tokenizer(sentence,
                                         add_special_tokens=False,
                                         truncation=False,
                                         return_attention_mask=False,
                                         return_tensors="pt")

		# Trying to understand what tokens are formed from the word
		# Extracting all the tokens
		tokens = self.tokenizer.tokenize(sentence)
		# Checks if the token is not equal to the original word
		# WARNING: WE ASSUME THE EMBEDDING_WORD_INDEX IS CORRECT!!!
		tokens_n: int = 1
		if tokens[self.embedding_word_index] != word:
			# print(f"Word <{word}> is not equal to token <{tokens[self.embedding_word_index]}>")
			for tok in tokens[(self.embedding_word_index + 1):]:
				if tok.startswith("##"):
					# If the token is a sub-word of the original word
					# print(f"\ttoken = {tok}")
					tokens_n += 1
				else:
					break
			# print("Total of tokens: ", tokens_n)
		# At the end, we obtain the indexes for the word tokens
		word_indexes = slice(self.embedding_word_index, self.embedding_word_index + tokens_n)
		# print(f"Tokens: {tokens[word_indexes]}")

		# We process the tokens with the BERT model
		embeddings = self.model(**tokens_encoding)

		# For now, the dimensions are:
		# [# all_layers, # batches, # tokens, # features]
		# but the first dimension (the <layers> one) is a Python list.
		# So we need to stack the elements in the list in one tensor:
		embeddings = torch.stack(embeddings.hidden_states, dim=0)
		# Now we remove the second dimension (batches) since it's unused
		embeddings = torch.squeeze(embeddings, dim=1)

		# We now have the layers as the first dimension:
		# [# all_layers, # tokens, # features]
		# We extract the selected layers
		if layers == "all":
			layers_embedding = embeddings
		else:
			if isinstance(layers, range):
				layers = list(layers)
			layers_embedding = embeddings[layers]

		# Now we permute dimensions, bringing the tokens dimension first:
		# From [# layers, # tokens, # features]
		# To [# tokens, # layers, # features]
		layers_embedding = layers_embedding.permute(1, 0, 2)

		# Now we extract the single embedding for the WORD tokens
		word_embedding = layers_embedding[word_indexes]
		if only_first_token:
			word_embedding = word_embedding[0]
		# The final tensor has dimensions: [# tokens, # desired_layers, # features]
		return word_embedding

	def embed_word_merged(self, word: str, layers: list[int] | range = "all") -> torch.Tensor:
		"""
		From a given word, returns its embeddings (for the desired layers) within the standard sentence-context.
		If the word is split into multiple tokens, the embeddings for the tokens are averaged.
		This assures only one token in return.

		:param word: The word to embed.
		:param layers: The desired layers of embeddings.
		:return: A matrix (PyTorch 2D-tensor) of dimensions [# layers, # features] for the embeddings.
		"""
		word_embedding = self.embed_word(word, layers, only_first_token=False)
		return torch.mean(word_embedding, dim=0)


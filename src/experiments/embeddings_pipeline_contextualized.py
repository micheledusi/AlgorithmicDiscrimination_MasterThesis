#########################################################################
#                            Dusi's Ph.D.                               #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# January 2023
# This script builds upon the "pipeline" experiment, but assessing the fact that
# single words maintain their original information despite the context they're inserted into.
# Put another way, we saw with a previous experiment that output embeddings for contextual words
# can be clustered based on words, and not based on contexts.
# This gives us faith in the fact that we can use contextualized embeddings in the pipeline experiment,
# renouncing to the "empty template" that was a bit tricky.

import numpy as np
import torch

from datasets import Dataset

import settings
from src.experiments.embeddings_contextual_template_analysis import TOKEN_WORD_IN_TMPL, TOKEN_WORD_TO_EMBED, \
	find_token_index_of_word, TOKEN_ARTICLE_IN_TMPL
from src.models.dimensionality_reducer import PipelineReducer, GenderClassifierReducer, TrainedPCAReducer
from src.models.gender_classifier import GenderLinearSupportVectorClassifier
from src.models.word_encoder import WordEncoder
from src.parsers.article_inference import infer_indefinite_article
from src.parsers.serializer import Serializer

EXPERIMENT_NAME: str = "embeddings_pipeline_contextualized"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_INPUT_DATA: str = settings.FOLDER_DATA + "/context_db"

# The input data are extracted from the folder "data/context_db/" and must be in the form of "TYPE_DOMAIN_ID.csv", where:
# - The type is either "words" or "templates", according to the csv content
# - The domain is the category of the words and templates. Ideally, each word-list has at least one domain-list, and vice-versa
# - The ID, to distinguish different lists within the same domain (e.g. job_1 and job_2)
EXPERIMENT_DOMAIN = "disciplines"
EXPERIMENT_WORDS_FILE = FOLDER_INPUT_DATA + "/" + "words_" + EXPERIMENT_DOMAIN + "_1.csv"
EXPERIMENT_TEMPLATES_FILE = FOLDER_INPUT_DATA + "/" + "templates_" + EXPERIMENT_DOMAIN + "_1.csv"

LAYERS: range = range(12, 13)
LAYERS_LABELS: list[str] = [f"{layer:02d}" for layer in LAYERS]
INTERMEDIATE_FEATURES_NUMBER: int = 50


def get_trained_pipeline() -> PipelineReducer:
	"""
	This method aggregates all the statements to obtain a working "dimensionality-reducing" pipeline,
	equals to the one contained in the paper.
	The method gets the gendered words from a previously saved database, then it trains a pipeline with:
	- A classifier-based layer, where N features are selected according to their weights
	- A PCA-based layer, where 2 features are computed according to their variance

	:return: The trained and ready pipeline
	"""
	# Retrieving gendered words to train pipelines
	ser = Serializer()
	gendered_ds = ser.load_dataset('gendered_words')
	train_x = np.asarray(gendered_ds['embedding'], dtype=np.float)[:, LAYERS]
	train_y = np.asarray(gendered_ds['gender'])
	print("Training embeddings shape: ", train_x.shape)

	lsvc_clf_768 = GenderLinearSupportVectorClassifier("base-lsvc-768", train_x, train_y, LAYERS_LABELS)
	reducer_768_n = GenderClassifierReducer(from_m=768, to_n=INTERMEDIATE_FEATURES_NUMBER, classifier=lsvc_clf_768)
	train_n = reducer_768_n.reduce(train_x)
	reducer_n_2 = TrainedPCAReducer(train_n, to_n=2)

	# Instancing the final reducer object
	reducer = PipelineReducer([
		reducer_768_n,
		reducer_n_2,
	])
	return reducer


def get_contextualized_embeddings() -> np.ndarray:
	"""
	This method aggregates all the actions required to obtain the embeddings.
	The embeddings refer to specific words, indicated by a constant naming the file they're in.
	Each word is "embedded" into a template (taken from a corresponding list) and then passed to BERT.
	The resulting embeddings for each word are averaged to obtain a single representation for each single word.
	The array of the results is returned.

	:return: A NumPy array of the averages of the embeddings for the examined words, through different templates.
	Dimension: [# words, # layers, # features]
	"""
	# Extracting words and templates list
	words_list: Dataset = Dataset.from_csv(EXPERIMENT_WORDS_FILE)
	templates_list: Dataset = Dataset.from_csv(EXPERIMENT_TEMPLATES_FILE)
	words_count: int = len(words_list)
	templates_count: int = len(templates_list)
	print(f"Extracted {words_count} words and {templates_count} templates.")

	# Data computation
	encoder: WordEncoder = WordEncoder()
	embs_list: list[torch.Tensor] = []
	# For each template
	for template in templates_list["template"]:
		# Preparing template
		template = template.replace(TOKEN_WORD_IN_TMPL, TOKEN_WORD_TO_EMBED)
		template = settings.TOKEN_CLS + " " + template + " " + settings.TOKEN_SEP
		# Find index of word "%s"
		index: int = find_token_index_of_word(encoder.tokenizer, TOKEN_WORD_TO_EMBED,
		                                      template.replace(TOKEN_ARTICLE_IN_TMPL, "y"))
		tmpl_embs_list: list[torch.Tensor] = []
		# For each word
		for word in words_list["word"]:
			article: str = infer_indefinite_article(word)
			encoder.set_embedding_template(template.replace(TOKEN_ARTICLE_IN_TMPL, article), word_index=index)
			# Computing the embedding of size: [#layers, #features]
			embedding: torch.Tensor = encoder.embed_word_merged(word, layers=LAYERS)
			tmpl_embs_list.append(embedding)

		# Coming to a single torch Tensor with size: [#words, #layers, #features]
		tmpl_embs: torch.Tensor = torch.stack(tmpl_embs_list)
		embs_list.append(tmpl_embs)

	# Coming to a single torch Tensor with size: [#templates, #words, #layers, #features]
	embs: torch.Tensor = torch.stack(embs_list)
	# Averaging the "templates" dimension, coming to a torch Tensor with size: [#words, #layers, #features]
	embs = torch.mean(embs, dim=0)
	return embs.cpu().detach().numpy()


def launch() -> None:
	print("Creating and training pipeline...")
	pipeline: PipelineReducer = get_trained_pipeline()

	print("Retrieving and computing input embeddings...")
	embeddings: np.ndarray = get_contextualized_embeddings()

	print("Reducing embeddings through pipeline...")
	reduced_embeddings = pipeline.reduce(embeddings)
	reduced_embeddings = np.squeeze(reduced_embeddings, axis=1)      # Removing the "Layers" dimension, assuming it's irrelevant (= one layer only)

	print(reduced_embeddings)



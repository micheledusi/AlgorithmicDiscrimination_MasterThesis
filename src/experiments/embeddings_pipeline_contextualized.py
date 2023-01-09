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
import gc

import numpy as np
import torch

from datasets import Dataset

import settings
from src.experiments.embeddings_contextual_template_analysis import TOKEN_WORD_TO_EMBED, find_token_index_of_word
from src.models.dimensionality_reducer import PipelineReducer, GenderClassifierReducer, TrainedPCAReducer
from src.models.gender_classifier import GenderLinearSupportVectorClassifier
from src.models.gender_mlm_regressor import MLMGenderRegressor
from src.models.word_encoder import WordEncoder
from src.parsers.article_inference import infer_indefinite_article
from src.parsers.serializer import Serializer
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter

EXPERIMENT_NAME: str = "embeddings_pipeline_contextualized"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_INPUT_DATA: str = settings.FOLDER_DATA + "/context_db"

# The input data are extracted from the folder "data/context_db/" and must be in the form of "TYPE_DOMAIN_ID.csv", where:
# - The type is either "words" or "templates", according to the csv content
# - The domain is the category of the words and templates. Ideally, each word-list has at least one domain-list, and vice-versa
# - The ID, to distinguish different lists within the same domain (e.g. job_1 and job_2)
EXPERIMENT_DOMAIN = "jobs"
WORDS_FILE_ID: int = 2
TEMPLATES_FILE_ID: int = 2
MLM_TEMPLATES_FILE_ID: int = 1
EXPERIMENT_WORDS_FILE = FOLDER_INPUT_DATA + f"/words/{EXPERIMENT_DOMAIN}_w{WORDS_FILE_ID}.csv"
EXPERIMENT_TEMPLATES_FILE = FOLDER_INPUT_DATA + f"/embs_templates/{EXPERIMENT_DOMAIN}_t{TEMPLATES_FILE_ID}.csv"
EXPERIMENT_MLM_TEMPLATES_FILE = FOLDER_INPUT_DATA + f"/mlm_templates/{EXPERIMENT_DOMAIN}_m{MLM_TEMPLATES_FILE_ID}.csv"

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


def get_contextualized_embeddings(words: Dataset, templates: Dataset) -> np.ndarray:
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
	words_count: int = len(words)
	templates_count: int = len(templates)
	print(f"Extracted {words_count} words and {templates_count} templates.")

	# Data computation
	encoder: WordEncoder = WordEncoder()
	embs_list: list[torch.Tensor] = []
	# For each template
	for template in templates["template"]:
		print(f"Processing template: \"{template}\"")
		# Preparing template
		template = template.replace(settings.TOKEN_WORD, TOKEN_WORD_TO_EMBED)
		template = settings.TOKEN_CLS + " " + template + " " + settings.TOKEN_SEP
		# Find index of word "%s"
		index: int = find_token_index_of_word(encoder.tokenizer, TOKEN_WORD_TO_EMBED,
		                                      template.replace(settings.TOKEN_ARTICLE, "y"))
		tmpl_embs_list: list[torch.Tensor] = []
		# For each word
		for word in words["word"]:
			article: str = infer_indefinite_article(word)
			encoder.set_embedding_template(template.replace(settings.TOKEN_ARTICLE, article), word_index=index)
			# Computing the embedding of size: [#layers, #features]
			embedding: torch.Tensor = encoder.embed_word_merged(word, layers=LAYERS)
			tmpl_embs_list.append(embedding.detach())

		# Cleaning cache
		gc.collect()
		torch.cuda.empty_cache()

		# Coming to a single torch Tensor with size: [#words, #layers, #features]
		tmpl_embs: torch.Tensor = torch.stack(tmpl_embs_list)
		embs_list.append(tmpl_embs)

	# Coming to a single torch Tensor with size: [#templates, #words, #layers, #features]
	embs: torch.Tensor = torch.stack(embs_list)
	# Averaging the "templates" dimension, coming to a torch Tensor with size: [#words, #layers, #features]
	embs = torch.mean(embs, dim=0)
	return embs.cpu().detach().numpy()


def compute_mlm_gender_polarization(words_list: list[str]) -> list[float]:
	mlm_templates_list: Dataset = Dataset.from_csv(EXPERIMENT_MLM_TEMPLATES_FILE)
	regressor: MLMGenderRegressor = MLMGenderRegressor(mlm_templates_list)
	return [regressor.predict_gender_polarization(word) for word in words_list]


def launch() -> None:
	ser: Serializer = Serializer()
	results_file: str = EXPERIMENT_DOMAIN + "-contextualized-pipeline-db"

	if ser.has_dataset(results_file):
		print("Loading results from file...")
		results = ser.load_dataset(results_file)
	else:
		print("Creating and training pipeline...")
		pipeline: PipelineReducer = get_trained_pipeline()

		print("Retrieving and computing input embeddings...")
		words_list: Dataset = Dataset.from_csv(EXPERIMENT_WORDS_FILE)
		templates_list: Dataset = Dataset.from_csv(EXPERIMENT_TEMPLATES_FILE)
		embeddings: np.ndarray = get_contextualized_embeddings(words=words_list, templates=templates_list)

		print("Reducing embeddings through pipeline...")
		reduced_embeddings = pipeline.reduce(embeddings)
		# Removing the "Layers" dimension, assuming it's irrelevant (= one layer only)
		reduced_embeddings = np.squeeze(reduced_embeddings, axis=1)
		# Bringing the features dimension to front, highlighting the "Xs" and the "Ys"
		reduced_embeddings = np.swapaxes(reduced_embeddings, 1, 0)

		print("Computing MLM gender polarization...")
		polarization: list[float] = compute_mlm_gender_polarization(words_list["word"])

		print("Packing results...")
		results = words_list \
			.add_column("x", reduced_embeddings[0]) \
			.add_column("y", reduced_embeddings[1]) \
			.add_column("mlm-polarization", polarization)
		ser.save_dataset(dataset=results, file_id=results_file)

		print("Saving results to CSV...")
		results.to_csv(
			path_or_buf=FOLDER_OUTPUT_TABLES + "/" + EXPERIMENT_DOMAIN + "-out." + settings.OUTPUT_TABLE_FILE_EXTENSION,
			index=None)

	# Plotting the data
	print("Plotting results on image chart...")
	embs: torch.Tensor = torch.stack([torch.tensor(results["x"]), torch.tensor(results["y"])])
	embs = torch.swapaxes(embs, 1, 0)
	plotter: EmbeddingsScatterPlotter = EmbeddingsScatterPlotter(embeddings=torch.tensor(embs))
	plotter.colormap = settings.COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE
	if len(results) < 30:
		plotter.labels = results["word"]
	plotter.colors = results["mlm-polarization"]
	plotter.plot_2d_pc()
	plotter.save(
		filename=FOLDER_OUTPUT_IMAGES + "/" + EXPERIMENT_DOMAIN + "-out." + settings.OUTPUT_IMAGE_FILE_EXTENSION)

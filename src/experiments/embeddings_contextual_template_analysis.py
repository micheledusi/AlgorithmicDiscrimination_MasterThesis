#########################################################################
#                             Dusi's Ph.D.                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import itertools

from datasets import Dataset
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score

from src.models.embeddings_clusterer import KMeansClusteringAlgorithm, ClusteringAlgorithm, \
	HierarchicalClusteringAlgorithm
from src.models.word_encoder import WordEncoder
from src.parsers.article_inference import infer_indefinite_article
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter
import settings

EXPERIMENT_NAME: str = "embeddings_contextual_template_analysis"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES
FOLDER_INPUT_DATA: str = settings.FOLDER_DATA + "/context_db"

# The input data are extracted from the folder "data/context_db/" and must be in the form of "TYPE_DOMAIN_ID.csv", where:
# - The type is either "words" or "templates", according to the csv content
# - The domain is the category of the words and templates. Ideally, each word-list has at least one domain-list, and vice-versa
# - The ID, to distinguish different lists within the same domain (e.g. job_1 and job_2)
EXPERIMENT_DOMAIN = "jobs"
WORDS_FILE_ID: int = 2
TEMPLATES_FILE_ID: int = 3
EXPERIMENT_WORDS_FILE = FOLDER_INPUT_DATA + f"/words/{EXPERIMENT_DOMAIN}_w{WORDS_FILE_ID}.csv"
EXPERIMENT_TEMPLATES_FILE = FOLDER_INPUT_DATA + f"/embs_templates/{EXPERIMENT_DOMAIN}_t{TEMPLATES_FILE_ID}.csv"

TOKEN_WORD_TO_EMBED: str = "%s"
TOKEN_AUX: str = "xx"  # A sequence of chars that (1) is NOT tokenized by BERT and (2) does NOT appear in sentences

# Defining the reference template
EMPTY_TEMPLATE: str = settings.TOKEN_CLS + " " + settings.TOKEN_WORD + " " + settings.TOKEN_SEP
BENCHMARK_TEMPLATE: str = EMPTY_TEMPLATE


def find_token_index_of_word(tokenizer, word: str, template: str) -> int:
	"""
	Given a BERT Tokenizer, a word and a sentence, it tokenizes the sentence with the BERT tokenizer, then it
	returns the index of the corresponding token that is equal to the word.
	If no corresponding tokens are found, this method raises an exception.

	:param tokenizer: The BERT Tokenizer that's used in the experiment
	:param word: The word we're looking for
	:param template: The sentence containing the word
	:return: An index, i.e. an integer number
	"""
	template.replace(word, TOKEN_AUX)
	tok_template = tokenizer.tokenize(template.replace(word, TOKEN_AUX))
	# print(f'Tokenized template: {tok_template}')
	index: int = tok_template.index(TOKEN_AUX)
	# print(f' => index: {index}')
	return index


def compute_cluster_mutual_information(db: Dataset, label_column: str, cluster_id_column: str) -> float:
	"""
	Given a Dataset, it considers two column as the original label and the labels given by the clustering.
	Then, it computes a metric of "mutual information", scoring the clustering quality between 0.0 and 1.0.
	Both columns must contain integers as labels and ids.

	:param db: The given dataset
	:param label_column: The column of the original labels (as integers)
	:param cluster_id_column: The column of the clustering labels (as integers)
	:return:
	"""
	return normalized_mutual_info_score(labels_true=db[label_column], labels_pred=db[cluster_id_column])


def launch() -> None:
	# Extracting words list
	words_list: Dataset = Dataset.from_csv(EXPERIMENT_WORDS_FILE)
	words_count: int = len(words_list)
	print(f"Extracted {words_count} words:")
	print(words_list["word"])

	# Extracting templates list
	templates_list: Dataset = Dataset.from_csv(EXPERIMENT_TEMPLATES_FILE)
	templates_count: int = len(templates_list)
	print(f"Extracted {templates_count} templates.")

	# Data computation

	encoder: WordEncoder = WordEncoder()
	embs_list: list[torch.Tensor] = []

	for template in templates_list["template"]:
		# print("Template: ", template)
		# Preparing template
		template = template.replace(settings.TOKEN_WORD, TOKEN_WORD_TO_EMBED)
		template = settings.TOKEN_CLS + " " + template + " " + settings.TOKEN_SEP

		# Find index of word "%s"
		index: int = find_token_index_of_word(encoder.tokenizer, TOKEN_WORD_TO_EMBED,
		                                      template.replace(settings.TOKEN_ARTICLE, "y"))
		tmpl_embs_list: list[torch.Tensor] = []

		for word in words_list["word"]:
			article: str = infer_indefinite_article(word)
			encoder.set_embedding_template(template.replace(settings.TOKEN_ARTICLE, article), word_index=index)

			# Computing the embedding of size: [#layers, #features]
			embedding: torch.Tensor = encoder.embed_word_merged(word, layers=[12])
			tmpl_embs_list.append(embedding)

		# Coming to a single torch Tensor with size: [#jobs, #layers, #features]
		tmpl_embs: torch.Tensor = torch.stack(tmpl_embs_list)
		embs_list.append(tmpl_embs)

	# Coming to a single torch Tensor with size: [#templates, #jobs, #layers, #features]
	embs: torch.Tensor = torch.stack(embs_list)
	# We remove the layer dimension, assuming it's only one layer
	embs = torch.squeeze(embs, dim=2)
	print()
	print("Final data size: ", embs.detach().size())
	print("                 [templates, words, features]")

	# Flattening the first two dimensions [#templates, #jobs] to one: [#samples]
	flatten = torch.nn.Flatten(0, 1)
	embs = flatten(embs)  # Obtaining a tensor of size [#samples, #features]
	embs_inputs = list(itertools.product(range(templates_count), range(
		words_count)))  # Obtaining a list of input pairs of integers [template_index, word_index]

	# Clustering algorithms
	# Defining the list
	clustering_algorithms: list[ClusteringAlgorithm] = [
		KMeansClusteringAlgorithm(num_clusters=templates_count, distance='cosine'),
		KMeansClusteringAlgorithm(num_clusters=words_count, distance='cosine'),
		HierarchicalClusteringAlgorithm(num_clusters=templates_count, distance='cosine', linkage='average'),
		HierarchicalClusteringAlgorithm(num_clusters=words_count, distance='cosine', linkage='average'),
		HierarchicalClusteringAlgorithm(num_clusters=templates_count, distance='euclidean', linkage='ward'),
		HierarchicalClusteringAlgorithm(num_clusters=words_count, distance='euclidean', linkage='ward'),
	]
	# Applying all the clustering algorithm to the embeddings "embs"
	clusters_ids: dict[str, torch.Tensor] = {str(algo): algo(embs) for (algo) in clustering_algorithms}

	# Data Logging

	# Creating a DB of all the results
	results_db: Dataset = Dataset.from_dict(mapping={"inputs": embs_inputs})

	def augment_db_row(sample):
		tmpl_id: int = sample["inputs"][0]
		word_id: int = sample["inputs"][1]
		sample["template_id"] = tmpl_id
		sample["word_id"] = word_id
		sample["template"] = templates_list["template"][tmpl_id]
		sample["word"] = words_list["word"][word_id]
		return sample

	results_db = results_db.map(function=augment_db_row)
	for (algo_name, clusters_labels) in clusters_ids.items():
		column_name: str = "label-" + algo_name
		# Adding columns to the database, one for each clustering algorithm
		results_db = results_db.add_column(name=column_name, column=clusters_labels.data.numpy())

		# Computing purity measures
		purity_score_templates: float = compute_cluster_mutual_information(results_db, "template_id", column_name)
		purity_score_words: float = compute_cluster_mutual_information(results_db, "word_id", column_name)

		# Logging to stdout
		print(f"\nClustering with algorithm: {algo_name}")
		print(f"\tNumber of resulting clusters: ", len(set(clusters_labels.data.numpy())))
		print(f"\tCluster similarity w.r.t {templates_count} templates: \t", purity_score_templates)
		print(f"\tCluster similarity w.r.t {words_count} words:     \t", purity_score_words)

	# Logging results database to CSV
	results_db.to_csv(FOLDER_OUTPUT_TABLES + "/out-" + EXPERIMENT_DOMAIN + "." + settings.OUTPUT_TABLE_FILE_EXTENSION)

	"""
	
	# Data visualization

	# Plotting the data
	plotter: EmbeddingsScatterPlotter = EmbeddingsScatterPlotter(embs)
	plotter.colormap = settings.COLORMAP_TAB_10_PALETTE
	plotter.colors = list(map(lambda pair: pair[1], embs_inputs))
	plotter.labels = list(map(lambda pair: f"{pair[0]}_{words_list['word'][pair[1]]}", embs_inputs))
	plotter.plot_2d_pc()
	plotter.save(
		filename=FOLDER_OUTPUT_IMAGES + "/out-" + EXPERIMENT_DOMAIN + "." + settings.OUTPUT_IMAGE_FILE_EXTENSION)
	# plotter.show()
	"""

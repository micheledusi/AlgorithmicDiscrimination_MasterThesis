#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import numpy as np
from src.models.word_encoder import WordEncoder
from src.models.embeddings_comparator import *
from src.parsers.occupations_parser import OccupationsParser
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import settings


templates_i: list[tuple[str, int]] = [
	("[CLS] i am a %s [SEP]", 4),
	("[CLS] i worked as a %s [SEP]", 5),
	("[CLS] i studied for years to become %s [SEP]", 7),
	("[CLS] finally i found a job as a %s [SEP]", 8),
	("[CLS] when i grow up i want to become a %s [SEP]", 10),
	("[CLS] someday i will be %s [SEP]", 5),
]

templates_you: list[tuple[str, int]] = [
	("[CLS] you are a %s [SEP]", 4),
	("[CLS] you worked as a %s [SEP]", 5),
	("[CLS] you studied for years to become %s [SEP]", 7),
	("[CLS] finally you found a job as a %s [SEP]", 8),
	("[CLS] when you grow up you want to become a %s [SEP]", 10),
	("[CLS] someday you will be %s [SEP]", 5),
]

templates_he: list[tuple[str, int]] = [
	("[CLS] he is a %s [SEP]", 4),
	("[CLS] he worked as a %s [SEP]", 5),
	("[CLS] he studied for years to become %s [SEP]", 7),
	("[CLS] finally he found a job as a %s [SEP]", 8),
	("[CLS] when he grows up he wants to become a %s [SEP]", 10),
	("[CLS] someday he will be %s [SEP]", 5),
]

templates_she: list[tuple[str, int]] = [
	("[CLS] she is a %s [SEP]", 4),
	("[CLS] she worked as a %s [SEP]", 5),
	("[CLS] she studied for years to become %s [SEP]", 7),
	("[CLS] finally she found a job as a %s [SEP]", 8),
	("[CLS] when she grows up he wants to become a %s [SEP]", 10),
	("[CLS] someday she will be %s [SEP]", 5),
]

templates_they: list[tuple[str, int]] = [
	("[CLS] they are a %s [SEP]", 4),
	("[CLS] they worked as a %s [SEP]", 5),
	("[CLS] they studied for years to become %s [SEP]", 7),
	("[CLS] finally they found a job as a %s [SEP]", 8),
	("[CLS] when they grow up they want to become a %s [SEP]", 10),
	("[CLS] someday they will be %s [SEP]", 5),
]

templates: dict = {
	"i":    templates_i,
	"you":  templates_you,
	"he":   templates_he,
	"she":  templates_she,
	"they": templates_they,
}


SELECTED_LAYERS: range = range(12, 13)


def compute_contextual_embedding(encoder: WordEncoder, word: str, layers) -> dict:
	"""
	Computes the embedding of a single word.
	The embedding is a dictionary with the pronouns as keys: "i", "you", "he", "she", "they".
	For each pronoun, a list of tensor is computed: one tensor for each template.
	The tensor is the result of the encoder model and comprehends all the selected layers in input to this function.
	:param encoder: The encoding model.
	:param word: The input word to the model
	:param layers: The selected layers to analyze
	:return: The dictionary of lists of tensors.
	"""
	embeddings: dict[str, list[torch.Tensor]] = {}
	for pron, templates_list in templates.items():
		pron_embeddings: list = []
		for tmpl, word_ix in templates_list:
			encoder.set_embedding_template(template=tmpl, word_index=word_ix)
			word_embedding = encoder.embed_word_merged(word, layers)
			pron_embeddings.append(word_embedding)
		embeddings[pron] = pron_embeddings
	return embeddings


def compute_contextual_embeddings_list(words: list[str]) -> dict:
	"""
	Computes the list of embeddings for each input word.
	:param words: The list of words
	:return: The dictionary associating words and embeddings
	"""
	encoder = WordEncoder()
	embeddings: dict[str, dict] = {}
	for w in words:
		word_embeddings = compute_contextual_embedding(encoder, w, SELECTED_LAYERS)
		embeddings[w] = word_embeddings
	return embeddings


def launch() -> None:
	# Extracting the list of occupations from WinoGender dataset
	parser = OccupationsParser()
	occs_list: list[str] = parser.occupations_list

	# Computing the embeddings
	occs_embs: dict = compute_contextual_embeddings_list(occs_list)

	measures = {}
	comparator = EmbeddingsComparator()
	# Pair euclidean distance
	comparator.add_metric(PairEuclideanDistance("he", "she"))
	comparator.add_metric(PairEuclideanDistance("he", "i"))
	comparator.add_metric(PairEuclideanDistance("she", "i"))
	comparator.add_metric(PairEuclideanDistance("he", "you"))
	comparator.add_metric(PairEuclideanDistance("she", "you"))
	comparator.add_metric(PairEuclideanDistance("he", "they"))
	comparator.add_metric(PairEuclideanDistance("she", "they"))
	comparator.add_metric(PairEuclideanDistance("i", "you"))
	# Sum of three euclidean distances
	comparator.add_metric(TripleEuclideanDistance("he", "she", "i"))
	comparator.add_metric(TripleEuclideanDistance("he", "she", "you"))
	comparator.add_metric(TripleEuclideanDistance("he", "she", "they"))
	comparator.add_metric(TripleEuclideanDistance("i", "you", "they"))
	# Euclidean distance from a center
	comparator.add_metric(EuclideanCenterDistance("he", "she", "i"))
	comparator.add_metric(EuclideanCenterDistance("he", "she", "you"))
	comparator.add_metric(EuclideanCenterDistance("he", "she", "they"))
	comparator.add_metric(EuclideanCenterDistance("i", "you", "they"))
	comparator.add_metric(EuclideanCenterDistance("he", "i", "you", "they"))
	comparator.add_metric(EuclideanCenterDistance("she", "i", "you", "they"))
	comparator.add_metric(EuclideanCenterDistance("he", "she", "i", "you", "they"))
	# Pair cosine similarity
	comparator.add_metric(PairCosineSimilarity("he", "she"))
	comparator.add_metric(PairCosineSimilarity("he", "i"))
	comparator.add_metric(PairCosineSimilarity("she", "i"))
	comparator.add_metric(PairCosineSimilarity("he", "you"))
	comparator.add_metric(PairCosineSimilarity("she", "you"))
	comparator.add_metric(PairCosineSimilarity("he", "they"))
	comparator.add_metric(PairCosineSimilarity("she", "they"))
	comparator.add_metric(PairCosineSimilarity("i", "you"))
	# Product of three cosine similarities
	comparator.add_metric(TripleCosineSimilarity("he", "she", "i"))
	comparator.add_metric(TripleCosineSimilarity("he", "she", "you"))
	comparator.add_metric(TripleCosineSimilarity("he", "she", "they"))
	comparator.add_metric(TripleCosineSimilarity("i", "you", "they"))

	# Computing and writing results
	elem_sep = settings.OUTPUT_TABLE_ARRAY_ELEM_SEPARATOR
	col_sep = settings.OUTPUT_TABLE_COL_SEPARATOR
	np.set_printoptions(precision=10)

	def print_tensor_array(t: torch.Tensor) -> str:
		# Prints a tensor with the desired format.
		t_np = t.detach().numpy()
		# If the tensor has two or more dimensions, it's printed in the standard way
		if len(t_np.shape) > 1:
			return t_np.__str__()
		# Else, if the tensor has one dimension and only one element
		elif t_np.shape[0] == 1:
			# The single element is printed without brackets
			return f"{t_np[0]}"
		# Else, if the tensor has one dimension and two or more elements (or even zero)
		else:
			s: str = "["
			s += elem_sep.join([str(e) for e in t_np])
			s += "]"
			return s

	with open(f"results/contextual_difference/metrics_lastlevel.{settings.OUTPUT_TABLE_FILE_EXTENSION}", "w") as f:
		header: str = f"word{col_sep}{comparator.names_header()}{col_sep}stat_bergsma{col_sep}stat_bls"
		print(header, file=f)
		for word, word_embs in occs_embs.items():
			print(f'Measuring metrics for word "{word}"...', end="")
			metrics = comparator(word_embs)
			print(word, end=col_sep, file=f)
			for m in metrics:
				print(print_tensor_array(m), end=col_sep, file=f)
			print(f"{parser.get_percentage(word, stat_name='bergsma')}{col_sep}{parser.get_percentage(word, stat_name='bls')}", file=f)
			print("Completed.")

	"""
	# Visualizing embeddings
	plotter = EmbeddingsScatterPlotter(embeddings)
	plotter.colors = [*([0] * occs_number), *([1] * occs_number)]
	plotter.labels = [*occ_list_m, *occ_list_f]
	plotter.plot_2d_pc()
	plotter.show()

	# Visualizing measures
	plt.figure()
	cmap = plt.get_cmap("viridis")
	norm = Normalize(vmin=0, vmax=100)

	pearson_x, pearson_y = [], []
	for occ, meas in measures.items():
		pct_dist: float = meas[0]
		emb_simi = meas[1].detach().numpy()
		emb_dist = meas[2].detach().numpy()
		plotted_measure = emb_simi

		pct_color = cmap(norm(pct_dist))
		plt.plot(selected_layers, plotted_measure, '.-', color=pct_color, label=occ)

		last_point = (selected_layers[-1], plotted_measure[-1])
		plt.annotate(occ, xy=last_point, xytext=(5, -2), textcoords="offset points")

		pearson_x.append(pct_dist)
		pearson_y.append(plotted_measure[-1])

	plt.xlabel("layers")
	plt.ylabel("cosine similarity")
	plt.title("Similarity between occupations in gender-opposite contexts")
	plt.show()

	# Correlation
	corr_tensor = torch.stack([torch.Tensor(pearson_x), torch.Tensor(pearson_y)])
	corr = torch.corrcoef(corr_tensor)[0][1]
	print("Correlation coefficient between real word disparity and measured cosine similarity: ", corr)
	"""
	return



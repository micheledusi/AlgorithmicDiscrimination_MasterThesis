#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a brief experiment about BERT spatial distribution of embeddings.
# To allow deterministic comparison between embeddings, the context is standardized to "[CLS] <word> [SEP]".
# Please notice that this is not how BERT should be used, but that's done only to obtain a single deterministic
# embedding for a given word / token.

import torch
from src.models.word_encoder import WordEncoder
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter
from src.parsers.occupations_parser import OccupationsParser

# Output path
OUTPUT_PATH = "results/embeddings_space"


def plot_occupations_embeddings(enc: WordEncoder) -> None:
	"""
	This functions plots all the occupations contained in the WinoGender dataset.
	Every profession is colored by its female percentage occupation: the female-occupied are magenta,
	the male-occupied are cyan.
	"""
	# Getting the occupations
	occ_parser = OccupationsParser()
	occ_pairs = occ_parser.get_sorted_female_occupations(stat_name="bls")
	# Unzipping list of pairs
	professions = [pair[0] for pair in occ_pairs]
	percentages = [pair[1] for pair in occ_pairs]

	# Computing embeddings for the last layer only
	professions_embeddings = [enc.embed_word(p, layers=[12]) for p in professions]
	professions_embeddings = torch.stack(professions_embeddings)

	print(professions_embeddings.size())

	# Plotting
	plotter = EmbeddingsScatterPlotter(professions_embeddings)
	plotter.labels = professions
	plotter.colors = percentages
	plotter.plot_2d_pc()
	plotter.save(f"{OUTPUT_PATH}/img/all_winogender_occupations_2D.png", timestamp=True)
	plotter.show()
	return


def plot_divisive_occupations_embeddings_history(enc: WordEncoder) -> None:
	"""
	This function plots a complex graph where the N most female-occupied and male-occupied professions are
	turned into embeddings and then plotted in a 2D space.
	Every embedding is plotted with its history along the 13 BERT layers.
	The female-occupied are magenta, the male-occupied are cyan.
	"""
	# Getting the occupations
	occ_parser = OccupationsParser()
	occupations_max = 5
	occ_pairs_f = occ_parser.get_sorted_female_occupations(max_length=occupations_max, female_percentage="highest")
	occ_pairs_m = occ_parser.get_sorted_female_occupations(max_length=occupations_max, female_percentage="lowest")
	occ_pairs = [*occ_pairs_f, *occ_pairs_m]

	# Unzipping list of pairs
	professions = [pair[0] for pair in occ_pairs]
	percentages = [pair[1] for pair in occ_pairs]

	# Computing embeddings
	professions_embeddings = [enc.embed_word(p) for p in professions]
	professions_embeddings = torch.stack(professions_embeddings)

	# Plotting
	plotter = EmbeddingsScatterPlotter(professions_embeddings)
	plotter.labels = professions
	plotter.colors = percentages
	plotter.plot_2d_pc()
	plotter.save(f"{OUTPUT_PATH}/img/extreme_winogender_occupations_history_2D.png", timestamp=True)
	plotter.show()
	return


def launch() -> None:
	"""
	The experiment launcher function.
	:return: None
	"""
	# Setup the word encoder with BERT
	enc = WordEncoder()

	plot_occupations_embeddings(enc)
	plot_divisive_occupations_embeddings_history(enc)
	return

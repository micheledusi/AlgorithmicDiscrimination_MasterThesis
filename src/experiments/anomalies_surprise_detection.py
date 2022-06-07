#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment is inspired by the paper "How is BERT surpised?".
# It aims to detect differences in likelihood of tokens between gender-declined paris of sentences.
# The un-likelihood of tokens is called "surprise". The average surprise of a sentence tokens is the sentence surprise.


import os.path
import pickle
import random
import sys
from libs.layerwise_anomaly.src import anomaly_model
from src.parsers.winogender_parser import get_sentences_pairs
from src.viewers.plot_heatmap_surprise import PairSurpriseHeatmapsPlotter


MODEL_SERIALIZED_FILE = "saved/models/anomaly_surprise_model.bin"
OUTPUT_IMG_FOLDER = "results/surprise_analysis/img"


def load_anomaly_model() -> anomaly_model.AnomalyModel:
	"""
	This function provide the Anomaly Model used in all the experiment of this script.
	An anomaly model can compute the "surprise" metric of tokens and sentences, given the embeddings produced by an
	encoder model such as BERT or RoBERTa.

	If a model already exists in a serialized binary file, that model is returned.
	Else, this function trains a new model from the BNC (British National Corpus) dataset, saves it and returns it.
	:return: The anomaly model as an object of class AnomalyModel.
	"""
	if os.path.exists(MODEL_SERIALIZED_FILE):
		print(f"\tExtracting model from file <{MODEL_SERIALIZED_FILE}>...", end="")
		with open(MODEL_SERIALIZED_FILE, "rb") as model_file:
			model = pickle.load(model_file)
		print("Completed.")
		return model
	else:
		print('\tLoading BNC (British National Corpus) sentences... ', end="")
		with open('data/bnc/bnc.pkl', 'rb') as f:
			# Loading sentences from pickle-serialized file
			# The serialized file has been obtained by running the original script of paper "How is BERT surprised?" on the BNC.
			bnc_sentences = pickle.load(f)
			# Randomly extracting N sentences (out of ~22k)
			random.seed(12345)
			bnc_sentences = random.sample(bnc_sentences, 10000)
		print("Completed.")

		print('\tTraining model with sentences...', end="")
		model = anomaly_model.AnomalyModel(
			bnc_sentences,
			encoder_name="bert-base-uncased",
			model_type="gmm",
			n_components=1,
			covariance_type="full",
			svm_kernel="rbf",
		)
		print("Completed.")

		print("\tSerializing model...", end="")
		with open(MODEL_SERIALIZED_FILE, "wb") as model_file:
			pickle.dump(model, model_file)
		print("Completed.")

		return model


def analyze_sentences_pairs(model: anomaly_model.AnomalyModel, sentence_pairs: list[tuple[str, str]], output_stream=sys.stdout):
	"""
	The first of two experiments contained in this script.
	This function analyzes a list of sentences pairs; for each sentence a global surprise is computed.
	Then, each sentence of each pair is compared with its associate to detect where the sentences' scores differ.

	The pairs are passed as a parameter from the outside, and come from the WinoGender dataset.
	Each sentence differs from the other by the gender of one pronoun.

	:param model: The AnomalyModel used to compute the surprise.
	:param sentence_pairs: The list of opposite-gender pairs.
	:param output_stream: The file out-stream where to save results in TSV format.
	:return: None
	"""
	# Computing the global surprise for sentences pair
	results = model.compute_sentences_pairs_list_surprise_merged(sentence_pairs)
	# <results> is a list of pair; each element of each pair is:
	# 1) associated with a sentence
	# 2) structured in 13 float, each one is the sentence's global surprise for the corresponding BERT layer

	last_layer = 12
	# Printing header
	print("score_male\tsent_male\tscore_female\tsent_female\tdifference", file=output_stream)
	for (sent_l, sent_r), (scores_l, scores_r) in zip(sentence_pairs, results):
		last_score_l = scores_l[last_layer]
		last_score_r = scores_r[last_layer]
		print(f"{last_score_l}\t{sent_l}", end="\t", file=output_stream)
		print(f"{last_score_r}\t{sent_r}", end="\t", file=output_stream)
		print(f"{last_score_l - last_score_r}", file=output_stream)
	return


def plot_sentences_pairs(model: anomaly_model.AnomalyModel, chosen_pairs: list[tuple[str, str]]):
	"""
	The second of two experiments contained in this script.
	Here we visualize the surprise for each tokens and layers of a couple of sentences.
	Like in the previous experiment, the two sentences differ from the gender of a single pronoun.

	The results are shown in three different heatmaps for each pair of sentences:
	- The first heatmap shows the surprise scores for the left sentence of the pair.
	- The third and last heatmap shows the same surprise score, but for the right sentence of the pair.
	- The middle heatmap plots the differences between the other two scores.

	:param model: The AnomalyModel used to compute the surprise.
	:param chosen_pairs: The list of opposite-gender pairs.
	:return: None
	"""
	for i, pair in enumerate(chosen_pairs):
		pair_result = model.compute_sentences_pair_surprise_per_tokens(pair)
		plotter = PairSurpriseHeatmapsPlotter(pair_result=pair_result)
		plotter.plot_surprise_heatmaps()
		plotter.save(f"{OUTPUT_IMG_FOLDER}/pair_surprise_heatmaps_{i}.png", timestamp=True)
	return


def launch() -> None:
	print('Loading anomaly model...', end="")
	model = load_anomaly_model()
	print("Completed.")

	# Reading sentences
	print("Reading and instantiating sentences from WinoGender dataset...", end="")
	sentence_pairs = get_sentences_pairs()
	print("Completed.")

	# [1] #
	print("Computing and printing results...", end="")
	nfile: str = "results/surprise_analysis/surprise_result.tsv"
	outfile = open(nfile, "w")
	analyze_sentences_pairs(model, sentence_pairs, output_stream=outfile)
	outfile.close()
	print("Completed.")

	# [2] #
	print("Analyzing single sentences...", end="")
	random.seed(12345)
	chosen_pairs = random.sample(sentence_pairs, 5)
	"""chosen_pairs = [
	 	("He is a doctor", "She is a doctor"),
	 	("He is a nurse", "She is a nurse"),
	 	("He is a mechanic", "She is a mechanic"),
	 	("He is a horse", "She is a horse"),
	 	("The doctor took his job seriously", "The doctor took her job seriously"),
	 	("The nurse took his job seriously", "The nurse took her job seriously"),
	 	("The engineer took his job seriously", "The engineer took her job seriously"),
	 	("The scientist took his job seriously", "The scientist took her job seriously"),
	 	("The librarian took his job seriously", "The librarian took her job seriously"),
	]"""
	plot_sentences_pairs(model, chosen_pairs)
	print("Completed.")

	pass

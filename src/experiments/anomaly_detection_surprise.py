#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment is inspired by the paper "How is BERT surprised?".
# It aims to detect differences in likelihood of tokens between gender-declined paris of sentences.
# The un-likelihood of tokens is called "surprise". The average surprise of a sentence tokens is the sentence surprise.


import os.path
import pickle
import random
import sys
from libs.layerwise_anomaly.src import anomaly_model
from src.parsers.winogender_templates_parser import get_sentences_pairs
from src.viewers.plot_heatmap_surprise import PairSurpriseHeatmapsPlotter
from src.parsers.winogender_occupations_parser import OccupationsParser
import settings

EXPERIMENT_NAME: str = "anomaly_detection_surprise"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES
MODEL_SERIALIZED_FILE: str = settings.FOLDER_SAVED_MODELS + "/anomaly_surprise_model.bin"

VISUALIZED_SENTENCES: int = 20
TRAINING_BNC_SENTENCES: int = 1000
PRINTED_TABLE_LAYERS: range = range(0, 13)


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
		with open(settings.FOLDER_DATA + '/bnc/bnc.pkl', 'rb') as f:
			# Loading sentences from pickle-serialized file
			# The serialized file has been obtained by running the original script of paper "How is BERT surprised?" on the BNC.
			bnc_sentences = pickle.load(f)
			# Randomly extracting N sentences (out of ~22k)
			random.seed(settings.RANDOM_SEED)
			bnc_sentences = random.sample(bnc_sentences, TRAINING_BNC_SENTENCES)
		print("Completed.")

		print('\tTraining model with sentences...', end="")
		model = anomaly_model.AnomalyModel(
			bnc_sentences,
			encoder_name=settings.DEFAULT_BERT_MODEL_NAME,
			model_type=settings.DEFAULT_DISTRIBUTION_MODEL_NAME,
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


def analyze_sentences_pairs(model: anomaly_model.AnomalyModel, sentence_pairs: list[tuple[str, str]],
                            output_stream=sys.stdout, row_ids: list[str] = None):
	"""
	The first of two experiments contained in this script.
	This function analyzes a list of sentences pairs; for each sentence a global surprise is computed.
	Then, each sentence of each pair is compared with its associate to detect where the sentences' scores differ.

	The pairs are passed as a parameter from the outside, and come from the WinoGender dataset.
	Each sentence differs from the other by the gender of one pronoun.

	:param model: The AnomalyModel used to compute the surprise.
	:param sentence_pairs: The list of opposite-gender pairs.
	:param output_stream: The file out-stream where to save results in TSV format.
	:param row_ids: If present, adds a row_id column
	:return: None
	"""
	# Computing the global surprise for sentences pair
	results = model.compute_sentences_pairs_list_surprise_merged(sentence_pairs)
	# <results> is a list of pair; each element of each pair is:
	# 1) associated with a sentence
	# 2) structured in 13 float, each one is the sentence's global surprise for the corresponding BERT layer

	# Printing header
	header_list: list[str] = []
	if row_ids is not None:
		header_list.append("row_id")
	for gender in ["m", "f"]:
		header_list.append(f"sentence_{gender}")
		for layer in PRINTED_TABLE_LAYERS:
			header_list.append(f"score_{gender}_{layer:02d}")
	for layer in PRINTED_TABLE_LAYERS:
		header_list.append(f"score_diff_{layer:02d}")
	header: str = settings.OUTPUT_TABLE_COL_SEPARATOR.join(header_list)
	print(header, file=output_stream)

	# Printing data
	for i in range(len(results)):
		data_list: list[str] = []
		if row_ids is not None:
			data_list.append(f"{row_ids[i]}")
		# Extracting current results
		(sent_m, sent_f) = sentence_pairs[i]
		(score_m, score_f) = results[i]
		# Appending male results
		data_list.append(sent_m)
		for layer in PRINTED_TABLE_LAYERS:
			data_list.append(f"{score_m[layer]:.8f}")
		# Appending female results
		data_list.append(sent_f)
		for layer in PRINTED_TABLE_LAYERS:
			data_list.append(f"{score_f[layer]:.8f}")
		# Appending differences by layer
		for layer in PRINTED_TABLE_LAYERS:
			data_list.append(f"{(score_m[layer] - score_f[layer]):.8f}")
		# Computing and printing the row string
		data_str: str = settings.OUTPUT_TABLE_COL_SEPARATOR.join(data_list)
		print(data_str, file=output_stream)

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
		plotter.save(f"{FOLDER_OUTPUT_IMAGES}/pair_surprise_heatmaps_{i}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}",
		             timestamp=False)
	return


def launch() -> None:
	print('Loading anomaly model...', end="")
	model = load_anomaly_model()
	print("Completed.")

	# Reading sentences
	print("Reading and instantiating sentences from WinoGender dataset...", end="")
	sentence_pairs: list[tuple[str, str]] = get_sentences_pairs()
	print("Completed.")

	# [1] #
	print("Computing and printing results...", end="")
	nfile: str = FOLDER_OUTPUT_TABLES + "/surprise_result." + settings.OUTPUT_TABLE_FILE_EXTENSION
	outfile = open(nfile, "w")
	analyze_sentences_pairs(model, sentence_pairs, output_stream=outfile)
	outfile.close()
	print("Completed.")

	# [2] #
	print("Analyzing single sentences...", end="")
	random.seed(settings.RANDOM_SEED)
	chosen_pairs = random.sample(sentence_pairs, VISUALIZED_SENTENCES)
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

	# [3] #
	print("Computing and printing results for WinoGender occupations...", end="")
	occ_parser = OccupationsParser()
	tmpl_m, tmpl_f = "he is a %s", "she is a %s"
	sentence_pairs = [(tmpl_m % occ, tmpl_f % occ) for occ in occ_parser.occupations_list]
	nfile: str = FOLDER_OUTPUT_TABLES + "/occupations_surprise_results." + settings.OUTPUT_TABLE_FILE_EXTENSION
	with open(nfile, "w") as outfile:
		analyze_sentences_pairs(model, sentence_pairs, output_stream=outfile, row_ids=occ_parser.occupations_list)
	print("Completed.")

	pass

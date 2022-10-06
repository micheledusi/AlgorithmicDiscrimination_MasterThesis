#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Compares gender classification with LinearSVCs and DecisionTrees.

import numpy as np

import settings
from src.experiments.embeddings_gender_subspace_detection import get_labeled_dataset, gendered_words
from src.models.gender_classifier import GenderLinearSupportVectorClassifier, GenderDecisionTreeClassifier, \
	_AbstractGenderClassifier
from src.models.gender_enum import Gender
from src.models.word_encoder import WordEncoder
from src.parsers import jobs_parser

EXPERIMENT_NAME: str = "embeddings_gender_classification_classifiers_comparison"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES


gendered_animal_words: dict[Gender, list[str]] = {
	# Gender.NEUTER: ["rabbit", "horse", "sheep", "pig", "chicken", "duck", "cattle", "goose", "fox", "tiger", "lion", ],
	Gender.MALE: ["buck", "stallion", "raw", "boar", "rooster", "drake", "bull", ],
	Gender.FEMALE: ["doe", "mare", "ewe", "sow", "hen", "duck", "cow", ],
}


def print_important_features_table(filepath: str, classifier: _AbstractGenderClassifier, layers_labels: list[str]) -> None:
	important_features = classifier.get_most_important_features()
	with open(filepath, "w") as f:
		header: list[str] = ['layer_label', 'feature_index', 'feature_importance']
		print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(header), file=f)
		# Printing rows
		for label, (indices, values) in zip(layers_labels, important_features):
			for index, value in zip(indices, values):
				row = [label, str(index), str(value)]
				print(settings.OUTPUT_TABLE_COL_SEPARATOR.join(row), file=f)
	return


def launch() -> None:
	encoder = WordEncoder()
	layers = range(0, 13)
	layers_labels = [f"{layer:02d}" for layer in layers]
	target_words = jobs_parser.get_words_list()

	train_x, train_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_words)
	eval_x, eval_y = get_labeled_dataset(encoder=encoder, layers=layers, data=gendered_animal_words)
	test_x, _ = get_labeled_dataset(encoder=encoder, layers=layers, data=target_words)

	clf_lsvc = GenderLinearSupportVectorClassifier(name='lsvc',
	                                               training_embeddings=np.asarray(train_x), training_genders=train_y,
	                                               layers_labels=layers_labels, print_summary=True)
	clf_tree = GenderDecisionTreeClassifier(name='tree',
	                                        training_embeddings=np.asarray(train_x), training_genders=train_y,
	                                        layers_labels=layers_labels, print_summary=True)

	print("Linear SVC:")
	clf_lsvc.evaluate(evaluation_embeddings=eval_x, evaluation_genders=eval_y)
	print("Decision Tree:")
	clf_tree.evaluate(evaluation_embeddings=eval_x, evaluation_genders=eval_y)

	# Printing the most important features for each layer of the classifier
	print_important_features_table(
		filepath=FOLDER_OUTPUT_TABLES + '/tree_model_important_feature.' + settings.OUTPUT_TABLE_FILE_EXTENSION,
		classifier=clf_tree, layers_labels=layers_labels)
	print_important_features_table(
		filepath=FOLDER_OUTPUT_TABLES + '/lsvc_model_important_feature.' + settings.OUTPUT_TABLE_FILE_EXTENSION,
		classifier=clf_lsvc, layers_labels=layers_labels)

	# Compare predictions
	predictions_lsvc = np.swapaxes(clf_lsvc.predict_gender_class(np.asarray(test_x)), 0, 1)
	predictions_tree = np.swapaxes(clf_tree.predict_gender_class(np.asarray(test_x)), 0, 1)
	# now predictions have dimensions: [# layers, # samples]

	for label, layer_predictions_lsvc, layer_predictions_tree in zip(layers_labels, predictions_lsvc, predictions_tree):
		print("Layer ", label)
		difference_count: int = 0
		for word, pred_lsvc, pred_tree in zip(target_words, layer_predictions_lsvc, layer_predictions_tree):
			if pred_lsvc != pred_tree:
				# print(f"\t'{word:20s}' has been predicted as:\t{pred_lsvc} by LSVC - {pred_tree} by TREE")
				difference_count += 1
		print(f"\tThe classifiers made different predictions in {difference_count} cases out of {len(target_words)}")
		print(f"\tAccuracy of the TreeClassifier with respect to the LinearSVClassifier: {1.0 - difference_count / len(target_words):.4%}")

	return

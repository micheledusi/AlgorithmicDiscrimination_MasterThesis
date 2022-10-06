#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains a brief experiment about BERT spatial distribution of embeddings.
# To allow deterministic comparison between embeddings, the context is standardized.
# Please notice that this is not how BERT should be used, but that's done only to obtain a single deterministic
# embedding for a given word / token.

import random

import torch

import settings
from src.experiments.embeddings_gender_subspace_detection import detect_gender_direction, get_trained_classifier
from src.experiments.mlm_gender_prediction_finetuned import prepare_sentences, train_group
from src.models.gender_classifier import GenderLinearSupportVectorClassifier
from src.models.trained_model_factory import TrainedModelForMaskedLMFactory
from src.models.word_encoder import WordEncoder
from src.parsers import jobs_parser

EXPERIMENT_NAME: str = "embeddings_gender_subspace_detection_finetuned"
FOLDER_OUTPUT: str = settings.FOLDER_RESULTS + "/" + EXPERIMENT_NAME
FOLDER_OUTPUT_IMAGES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_IMAGES
FOLDER_OUTPUT_TABLES: str = FOLDER_OUTPUT + "/" + settings.FOLDER_TABLES
FOLDER_SAVED_MODELS_EXPERIMENT: str = settings.FOLDER_SAVED_MODELS + "/" + EXPERIMENT_NAME

LAYERS: range | list = range(12, 13)
LAYERS_LABELS: list[str] = [f"{layer:02d}" for layer in LAYERS]


def launch() -> None:
	# Templates group
	train_occs_list: list[str] = jobs_parser.get_words_list()
	sentences: list[str] = prepare_sentences(templates_group=train_group, occupations=train_occs_list)
	print("Total number of sentences: ", len(sentences))

	# Sampling a subset of sentences
	random.seed(settings.RANDOM_SEED)
	model_name = settings.DEFAULT_BERT_MODEL_NAME
	factory = TrainedModelForMaskedLMFactory(model_name=model_name)
	training_samples: list[int] = [0, 500, 1000, 2000, 5000, 10000, 20000]

	# For every number of training dataset size
	for samples_number in training_samples:
		sentences_sampled = random.sample(sentences, samples_number)
		saved_model_ft_path = FOLDER_SAVED_MODELS_EXPERIMENT + f"/gender_subspace_detection_{model_name}_{samples_number}"
		model = factory.get_model(fine_tuning_text=sentences_sampled, load_or_save_path=saved_model_ft_path)
		model.to(settings.pt_device)
		encoder = WordEncoder(tokenizer=factory.tokenizer, model=model)

		# Detecting the gender direction
		clf = get_trained_classifier(classifier_class=GenderLinearSupportVectorClassifier,
		                             model_name=f"trained-{samples_number}", encoder=encoder,
		                             layers=LAYERS, layers_labels=LAYERS_LABELS)
		detect_gender_direction(classifier=clf, encoder=encoder, layers=LAYERS, layers_labels=LAYERS_LABELS,
		                        folder_output_images=FOLDER_OUTPUT_IMAGES, folder_output_tables=FOLDER_OUTPUT_TABLES)

		# Cleaning CUDA cache
		torch.cuda.empty_cache()
	return

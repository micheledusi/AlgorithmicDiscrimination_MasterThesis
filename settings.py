#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This file contains the settings for the whole project, in a centralized place.


import numpy as np
import torch

from PIL import ImageColor
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from src.models.gender_enum import Gender

# PYTORCH COMPUTING

# Determinism

RANDOM_SEED: int = 42
torch.manual_seed(RANDOM_SEED)

# If available, torch computes on a parallel architecture
pt_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Machine Learning
TRAIN_TEST_SPLIT_PERCENTAGE = 0.2

# FILES

# Folders structure
FOLDER_DATA = 'data'
FOLDER_RESULTS = 'results'
FOLDER_SAVED = 'saved'
FOLDER_SAVED_MODELS = FOLDER_SAVED + '/models'
FOLDER_SAVED_DATA = FOLDER_SAVED + '/data'

FOLDER_TABLES = 'tables'
FOLDER_IMAGES = 'img'

# Serialization
OUTPUT_SERIALIZED_FILE_EXTENSION: str = 'pkl'

# ENCODING

# Distribution models
DISTRIBUTION_GAUSSIAN_MIXTURE_MODEL_NAME: str = 'gmm'
DISTRIBUTION_SUPPORT_VECTOR_MACHINE_NAME: str = 'svm'
DEFAULT_DISTRIBUTION_MODEL_NAME: str = DISTRIBUTION_GAUSSIAN_MIXTURE_MODEL_NAME

# Model names and parameters
DEFAULT_BERT_MODEL_NAME: str = 'bert-base-uncased'

# Templates
TOKEN_CLS: str = '[CLS]'
TOKEN_SEP: str = '[SEP]'
TOKEN_MASK: str = '[MASK]'
DEFAULT_STANDARDIZED_EMBEDDING_TEMPLATE: str = TOKEN_CLS + " %s " + TOKEN_SEP
DEFAULT_STANDARDIZED_EMBEDDING_WORD_INDEX: int = 1

# PRINTING

# Printing things in tables
OUTPUT_TABLE_COL_SEPARATOR: str = '\t'
OUTPUT_TABLE_ARRAY_ELEM_SEPARATOR: str = ' '
OUTPUT_TABLE_FILE_EXTENSION: str = 'tsv'

# PLOTTING

# PyPlot colormaps

OUTPUT_IMAGE_FILE_EXTENSION: str = 'png'

RGBA_MALE = ImageColor.getcolor(Gender.MALE.color, 'RGBA')
RGBA_FEMALE = ImageColor.getcolor(Gender.FEMALE.color, 'RGBA')
RGBA_NEUTER = ImageColor.getcolor(Gender.NEUTER.color, 'RGBA')
RGBA_MALE_TRANSPARENT = [*RGBA_MALE[:3], 0]
RGBA_FEMALE_TRANSPARENT = [*RGBA_FEMALE[:3], 0]
RGBA_NEUTER_TRANSPARENT = [*RGBA_NEUTER[:3], 0]
RGBA_WHITE_TRANSPARENT = [255, 255, 255, 0]

COLORMAP_GENDER_MALE2FEMALE: ListedColormap = LinearSegmentedColormap.from_list(
	name='male-female',
	colors=np.asarray([RGBA_MALE, RGBA_FEMALE]) / 255)
COLORMAP_GENDER_MALE2NEUTER2FEMALE: ListedColormap = LinearSegmentedColormap.from_list(
	name='male-neuter-female',
	colors=np.asarray([RGBA_MALE, RGBA_NEUTER, RGBA_FEMALE]) / 255)
COLORMAP_GENDER_MALE2TRANSPARENT2FEMALE: ListedColormap = LinearSegmentedColormap.from_list(
	name='male-transparent-female',
	colors=np.asarray([RGBA_MALE, RGBA_WHITE_TRANSPARENT, RGBA_FEMALE]) / 255)
COLORMAP_GENDER_MALE25_TRANSPARENT50_FEMALE25: ListedColormap = LinearSegmentedColormap.from_list(
	name='male-transparent-female',
	colors=np.asarray([RGBA_MALE,                   # __0 %
	                   RGBA_MALE_TRANSPARENT,       # _25 %
	                   RGBA_WHITE_TRANSPARENT,      # _50 %
	                   RGBA_FEMALE_TRANSPARENT,     # _75 %
	                   RGBA_FEMALE                  # 100 %
	                   ]) / 255)

COLORMAP_NAME_GENDER_CYAN2PINK: str = 'cool'
COLORMAP_NAME_PALETTE: str = 'Set2'

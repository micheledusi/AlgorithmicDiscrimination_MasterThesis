#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This file contains the settings for the whole project, in a centralized place.

# DETERMINISM

# Random seed
RANDOM_SEED: int = 42

# FILES

# Folders structure
FOLDER_DATA = 'data'

# ENCODING

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
GENDER_CYAN2PINK_COLORMAP_NAME: str = 'cool'

OUTPUT_IMAGE_FILE_EXTENSION: str = 'png'

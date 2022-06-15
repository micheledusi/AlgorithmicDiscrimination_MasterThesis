#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import settings
from src.experiments import embeddings_spatial_analysis
from src.experiments import anomalies_surprise_detection
from src.experiments import gender_subspace_detection
from src.experiments import gendered_context_difference_measuring
from src.experiments import gender_prediction
import torch

if __name__ == '__main__':
    torch.manual_seed(settings.RANDOM_SEED)

    # embeddings_spatial_analysis.launch()
    # anomalies_surprise_detection.launch()
    # gender_subspace_detection.launch()
    # gendered_context_difference_measuring.launch()
    gender_prediction.launch()
    pass

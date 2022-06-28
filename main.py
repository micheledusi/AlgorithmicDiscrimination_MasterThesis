#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

import settings
from src.experiments import embeddings_spatial_analysis
from src.experiments import anomalies_surprise_detection
from src.experiments import gender_subspace_detection
from src.experiments import gendered_context_difference_measuring
from src.experiments import mlm_gender_prediction
from src.experiments import mlm_gender_prediction_finetuned
from src.experiments import embeddings_gender_classification

if __name__ == '__main__':
    # embeddings_spatial_analysis.launch()
    gender_subspace_detection.launch()
    # anomalies_surprise_detection.launch()
    # gendered_context_difference_measuring.launch()
    # mlm_gender_prediction.launch()
    # mlm_gender_prediction_finetuned.launch()
    # embeddings_gender_classification.launch()
    pass

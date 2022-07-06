#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

from src.experiments import embeddings_static_analysis
from src.experiments import embeddings_gender_subspace_detection
from src.experiments import embeddings_contextual_analysis
from src.experiments import embeddings_gender_classification
from src.experiments import anomaly_detection_surprise
from src.experiments import mlm_gender_prediction
from src.experiments import mlm_gender_prediction_finetuned
from src.experiments import mlm_gender_perplexity

if __name__ == '__main__':
    # Spatial analysis of embeddings
    # embeddings_static_analysis.launch()
    # embeddings_contextual_analysis.launch()
    # embeddings_gender_classification.launch()
    # embeddings_gender_subspace_detection.launch()

    # Anomaly Detection
    # anomaly_detection_surprise.launch()

    # Masked Language Modeling
    # mlm_gender_prediction.launch()
    # mlm_gender_prediction_finetuned.launch()
    mlm_gender_perplexity.launch()

    pass

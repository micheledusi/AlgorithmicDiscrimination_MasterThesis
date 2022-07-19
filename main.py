#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

from src.experiments import embeddings_static_analysis
from src.experiments import embeddings_gender_subspace_detection_pca
from src.experiments import embeddings_gender_subspace_detection
from src.experiments import embeddings_gender_subspace_detection_finetuned
from src.experiments import embeddings_contextual_analysis
from src.experiments import embeddings_gender_classification_contextual
from src.experiments import embeddings_gender_classification_classifiers_comparison
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
    # embeddings_gender_subspace_detection_finetuned.launch()
    embeddings_gender_subspace_detection_pca.launch()
    # embeddings_gender_classification_classifiers_comparison.launch()

    # Anomaly Detection
    # anomaly_detection_surprise.launch()

    # Masked Language Modeling
    # mlm_gender_prediction_finetuned.launch()
    # mlm_gender_prediction_finetuned.launch()
    # mlm_gender_perplexity.launch()

    pass

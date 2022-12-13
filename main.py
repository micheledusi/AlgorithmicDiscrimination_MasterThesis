#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

from src.experiments import embeddings_static_analysis
from src.experiments import embeddings_gender_subspace_detection_pca
from src.experiments import embeddings_gender_subspace_detection
from src.experiments import embeddings_gender_subspace_detection_finetuned
from src.experiments import embeddings_contextual_analysis
from src.experiments import embeddings_contextual_template_analysis
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
    embeddings_contextual_template_analysis.launch()

    # CLASSIFICAZIONE
    # embeddings_gender_subspace_detection.launch()
    # embeddings_gender_subspace_detection_finetuned.launch()
    # embeddings_gender_classification_contextual.launch()
    # embeddings_gender_classification_classifiers_comparison.launch()

    # SUPERVISIONATI / GRAFICI / FEATURES-BASED
    # embeddings_gender_subspace_detection_pca.launch()

    # ANOMALY DETECTION
    # anomaly_detection_surprise.launch()

    # MASKED LANGUAGE MODELING
    # mlm_gender_prediction.launch()
    # mlm_gender_prediction_finetuned.launch()
    # mlm_gender_perplexity.launch()

    pass

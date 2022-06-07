#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# In the <experiments> folder there are scripts implementing the various trials of this thesis:
# - Embedding Spatial Analysis: a study of the geometry of embeddings
# - Anomalies Surprise Analysis: an attempt to detect anomalies in gender-declined sentences pairs

from src.experiments import embeddings_spatial_analysis
from src.experiments import anomalies_surprise_detection
from src.experiments import gender_subspace_detection

if __name__ == '__main__':
    # embeddings_spatial_analysis.launch()
    anomalies_surprise_detection.launch()
    # gender_subspace_detection.launch()
    pass

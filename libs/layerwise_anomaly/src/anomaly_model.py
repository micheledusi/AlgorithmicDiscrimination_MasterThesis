############
# Code from: https://github.com/SPOClab-ca/layerwise-anomaly
# Li, B., Zhu, Z., Thomas, G., Xu, Y., and Rudzicz, F. (2021)
# How is BERT surprised? Layerwise detection of linguistic anomalies.
# In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL).
############

import sklearn.mixture
import numpy as np
from numpy import ndarray

from libs.layerwise_anomaly.src.sentence_encoder import SentenceEncoder

DEFAULT_ENCODER_NAME: str = "roberta-base"
DEFAULT_MODEL_TYPE: str = "gmm"


class AnomalyModel:
    """Model that uses GMM on embeddings generated by BERT for finding syntactic
    or semantic anomalies.
    """

    def __init__(self, train_sentences,
                 encoder_name: str = DEFAULT_ENCODER_NAME,
                 model_type: str = DEFAULT_MODEL_TYPE,
                 # Parameters for GMM model type:
                 n_components: int = 1,
                 covariance_type='full',
                 # Parameters for SVM model type:
                 svm_kernel='rbf'
                 ):
        self.enc = SentenceEncoder(model_name=encoder_name)
        self.gmms = []

        # Assumes base models have 12+1 layers, large models have 24+1
        self.num_encoder_layers = 25 if 'large' in encoder_name else 13

        _, all_vecs = self.enc.contextual_token_vecs(train_sentences)
        # <all_vecs> is the list of numpy tensor, one for each sentence
        # One sentence => np.array(sentence length, 13, 768)

        for layer in range(self.num_encoder_layers):
            sent_vecs = np.vstack([vs[:, layer, :] for vs in all_vecs])

            gmm = None
            if model_type == 'gmm':
                # GMM = Gaussian Mixture Model
                gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            elif model_type == 'svm':
                # SVM = Support Vector Machine
                gmm = sklearn.svm.OneClassSVM(kernel=svm_kernel)

            gmm.fit(sent_vecs)
            # After training, the GMM is appended to a list of GMMs, stored into the object "AnomalyModel"
            # Basically, we have a GMM for each layer of the encoder model
            # (e.g. for BERT, => 13 GMMs)
            self.gmms.append(gmm)

    def compute_sentence_surprise_per_tokens(self, sentence: str) -> tuple[list, np.ndarray]:
        """
        Computes the surprise for a single sentence.
        :param sentence: The sentence to analyze.
        :return: The scores as a NumPy ndarray of dimensions (#layers, #tokens)
        """
        # Here I did a bad thing: I pass to the encoder a list with a single item, and I take the single item returned.
        # That's because the encoder method works with list, but I'm working on a single sentence here
        tokens, vecs = self.enc.contextual_token_vecs([sentence])
        tokens: list = tokens[0]
        vecs: ndarray = vecs[0]
        assert len(tokens) == vecs.shape[0]

        sentence_scores: np.ndarray = np.zeros(shape=(self.num_encoder_layers, len(tokens)))
        # For each layer of the encoder model
        for layer in range(self.num_encoder_layers):
            current_gmm = self.gmms[layer]
            # Extracting the embeddings for each token, but for one single layer
            embeddings = vecs[:, layer, :]
            # Scoring the tokens via their embeddings, for the current layer
            layer_scores = current_gmm.score_samples(embeddings)
            # Returns an array of (#tokens) = (#samples) scores
            sentence_scores[layer] = layer_scores

        # In the end, the sentence_scores are returned
        return tokens, sentence_scores

    def compute_sentences_pair_surprise_per_tokens(self, pair: tuple[str, str]) -> dict:
        """
        Computes the surprise for a couple of sentences and the difference between the two.
        :return: a dictionary with:
            - The tokens as a pair, for the left and right sentence
            - The scores as a tuple (first_sentence_scores, second_sentence_scores) where:
                each item is a NumPy ndarray of dimensions (#layers, #tokens)
            - The difference between the scores
        """
        left_sentence = pair[0]
        right_sentence = pair[1]
        tokens_l, scores_l = self.compute_sentence_surprise_per_tokens(left_sentence)
        tokens_r, scores_r = self.compute_sentence_surprise_per_tokens(right_sentence)
        scores_diff = np.abs(scores_l - scores_r)
        result = {
            "tokens": (tokens_l, tokens_r),
            "scores": (scores_l, scores_r),
            "difference": scores_diff,
        }
        return result

    def compute_sentences_list_surprise_per_tokens(self, sentences_list: list[str]):
        """
        Given a list of sentences, computes the surprise for each token of each sentence.
        :param sentences_list: The list of sentences to evaluate
        :return: (all_tokens, all_scores), where
            all_tokens is List[List[token]]
            all_scores is List[np.array(#layers, #tokens)]
        """
        # Extracting the tokens from the sentences, already encoded
        all_tokens, all_vecs = self.enc.contextual_token_vecs(sentences_list)
        #   all_tokens  is a List[List[tokens]], one list for each sentence.
        #   all_vecs    is a List[np.array(sentence length, 13, 768)], one array for each sentence.
        all_scores = []

        for sent_ix in range(len(sentences_list)):
            # <sent_ix> is the index
            # Now we're getting the tokens and embeddings for a specific sentence
            tokens = all_tokens[sent_ix]            # is a List[token]
            vecs = all_vecs[sent_ix]                # is a np.array(sentence length, 13, 768)
            assert len(tokens) == vecs.shape[0]     # Asserting the #tokens == sentence length

            sentence_scores = []
            # For each layer of the encoder model
            for layer in range(self.num_encoder_layers):
                # Extracting the current GMM
                current_gmm = self.gmms[layer]
                # Extracting the embeddings for each token, but for one single layer
                embeddings = vecs[:, layer, :]
                # Scoring the tokens via their embeddings, for the current layer
                # The method <score_samples> computes the log-likelihood of each sample of the list
                # The input list is an array of shape (#samples, #features), i.e. a list of (embeddings)
                layer_scores = current_gmm.score_samples(embeddings)
                # Returns an array of (#samples) scores
                sentence_scores.append(layer_scores)

            # In the end, the sentence scores are appended in the all_scores array
            all_scores.append(np.array(sentence_scores))

        return all_tokens, all_scores

    @staticmethod
    def unzip(pairs_list: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        unzipped_obj = list(zip(*pairs_list))
        return list(unzipped_obj[0]), list(unzipped_obj[1])

    def compute_sentence_pairs_list_surprise_per_tokens(self, sentences_pairs: list[(str, str)]):
        """
        Evaluate surprise for each token of sentence pairs.
        Sentences in pair are called (left, right).
        :param sentences_pairs: The sentences pairs to evaluate, a list[(str, str)]
        :return: The list of scores pairs: List[(scores for the left sentence, scores for the right sentence)]
        """
        # Unzipping the list of pairs
        left_sentences, right_sentences = AnomalyModel.unzip(sentences_pairs)
        # Computing surprise scores
        left_scores_per_tokens = self.compute_sentences_list_surprise_per_tokens(left_sentences)[1]
        right_scores_per_tokens = self.compute_sentences_list_surprise_per_tokens(right_sentences)[1]
        # Re-zipping the surprise scores
        pairs_scores = list(zip(left_scores_per_tokens, right_scores_per_tokens))
        # This is quite stupid, but it's due to the previous structure of the code...
        return pairs_scores

    def compute_sentences_pairs_list_surprise_merged(self, sentences_pairs: list[(str, str)]):
        """
        Evaluate global surprise for each sentence of a pair, for each pair of a list.
        :param sentences_pairs: The sentences pairs to evaluate, a list[(str, str)]
        :return: The list of scores pairs: List[(score for the left sentence, score for the right sentence)]
        """
        # Unzipping the list of pairs
        left_sentences, right_sentences = AnomalyModel.unzip(sentences_pairs)
        # Computing surprise scores per tokens
        left_scores_per_tokens = self.compute_sentences_list_surprise_per_tokens(left_sentences)[1]
        right_scores_per_tokens = self.compute_sentences_list_surprise_per_tokens(right_sentences)[1]
        # Summing scores of each token and dividing by the number of tokens
        left_scores_per_sentence = [np.average(sentence_scores, axis=1) for sentence_scores in left_scores_per_tokens]
        right_scores_per_sentence = [np.average(sentence_scores, axis=1) for sentence_scores in right_scores_per_tokens]
        # Re-zipping the surprise scores
        pairs_scores: list[tuple[ndarray, ndarray]] = list(zip(left_scores_per_sentence, right_scores_per_sentence))
        return pairs_scores
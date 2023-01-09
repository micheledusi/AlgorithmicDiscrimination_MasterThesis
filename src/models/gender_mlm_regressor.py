#########################################################################
#                            Dusi's Ph.D.                               #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This module implements a minimal interface for the gender bias analysis through the MLM technique.
# The regressor object takes a word and return its gender polarization (negative for male gender, positive for female gender).
# At the moment, in order to simplify the analysis and follow the implementation of the most common models, we're only
# considering two genders. We're conscious this is not the reality, and we're aiming to consider different possibilities
# in the near future.
# From the gender polarization the bias can be obtained by applying the "absolute value" function.

import numpy as np
from datasets import Dataset
from transformers import pipeline

import settings
from src.models.gender_enum import Gender
from src.models.trained_model_factory import TrainedModelForMaskedLMFactory
from src.parsers.article_inference import infer_indefinite_article


class MLMGenderRegressor:
	"""
	Simple class for a gender-polarization predictor which uses MLM.
	It requires some templates to analyze the inquired words. Such templates must contain:
	- The string "[MASK]", which will be replaced by two subject pronouns: <he> and <she>.
	- The string "[WORD]", which will be replaced by the inquired word
	- (optional) The string "[ART]", which will be replaced by the article accorded to the inquired word.

	It is programmer's duty to assert that each word replacement leads to a correct sentence. No check is done by the class.

	Other eventual tokens or placeholders will NOT be replaced (such as possessive pronouns).
	"""

	def __init__(self, templates: Dataset | list[str]):
		if isinstance(templates, Dataset):
			if "template" not in templates.column_names:
				raise Exception("Cannot use templates dataset without a \"template\" column. Please, provide such column in the input dataset.")
			templates = templates["template"]   # Extracting strings list from Dataset
		self.__templates: list[str] = templates

		# Instancing the pipeline for MLM
		model_name = settings.DEFAULT_BERT_MODEL_NAME
		factory = TrainedModelForMaskedLMFactory(model_name=model_name)
		base_model = factory.get_model(fine_tuning_text=None)
		self.__unmasker = pipeline("fill-mask", model=base_model,
		                           tokenizer=factory.tokenizer,
		                           targets=[Gender.FEMALE.nom_pronoun, Gender.MALE.nom_pronoun],
		                           top_k=2,
		                           device=0)

	def predict_gender_polarization(self, word: str) -> float:
		scores: list[float] = []
		for template in self.__templates:
			article: str = infer_indefinite_article(word)
			masked_sentence = template.replace(settings.TOKEN_WORD, word) \
				.replace(settings.TOKEN_ARTICLE, article)
			# Computing the results
			results = self.__unmasker(masked_sentence)  # <Results> is an iterable of results, each one associated with a gender
			results_dict: dict = {res["token_str"]: res["score"] for res in results}
			template_polarization = results_dict[Gender.FEMALE.nom_pronoun] - results_dict[Gender.MALE.nom_pronoun]
			scores.append(template_polarization)
		# Computing the mean overall the templates' scores
		score = float(np.mean(scores))
		return score


if __name__ == "__main__":
	tmpls = [
		"[MASK]'s very happy to be [ART] [WORD].",
		"[MASK]'s [ART] [WORD].",
		"[MASK] will soon be [ART] [WORD]",
	]
	regressor = MLMGenderRegressor(tmpls)
	for w in ["plumber", "nurse", "carpenter", "waitress", "doctor"]:
		s: float = regressor.predict_gender_polarization(w)
		print(f"Score for word: {w:20s}: {s:2.8f}")


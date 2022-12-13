#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Inferring the article for english names.


AN_PREFIXES: tuple[str, ...] = ('a', 'e', 'i', 'o', "un")
A_PREFIXES: tuple[str, ...] = ('ow', 'uni', 'eu')


def infer_indefinite_article(noun: str) -> str:
	"""
	Returns the indefinite article for the give noun based on simplistic assumptions.
	:param noun: A give noun
	:return: The corresponding indefinite article ('a'/'an'), based on some assumptions
	"""
	if noun.startswith(AN_PREFIXES) and not noun.startswith(A_PREFIXES):
		return 'an'
	return 'a'


def get_articles_list(nouns: list[str]) -> list[str]:
	"""
	:param nouns: A list of nouns.
	:return: The list of articles (without the nouns!).
	"""
	return [infer_indefinite_article(w) for w in nouns]


def add_articles(nouns: list[str]) -> list[str]:
	"""
	:param nouns: A list of nouns.
	:return: The list of nouns with an article before each of them.
	"""
	return [infer_indefinite_article(w) + ' ' + w for w in nouns]

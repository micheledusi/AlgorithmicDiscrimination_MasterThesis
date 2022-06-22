#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Python classes for a generic string with a replaceable token.

from src.models.gender_enum import Gender


class Template:
	sentence: str
	targets: list[str] | None

	def __init__(self, sentence: str, targets: list[str] | None = None):
		self.sentence = sentence
		self.targets = targets

	def instance_with(self, word: str, mask_token: str = '%s') -> str:
		return self.sentence.replace(mask_token, word)


class TemplatesGroup:
	name: str | None
	__templates: list[Template]
	__targets: set[str] = {}
	targets_by_gender: dict[Gender, list[str]]

	def __init__(self, name: str = None):
		self.name = name

	@property
	def templates(self) -> list[Template]:
		return self.__templates

	@templates.setter
	def templates(self, templates: list[Template]) -> None:
		self.__templates = templates
		# Updates all the templates with empty targets, with the group targets
		for template in self.__templates:
			if template.targets is None or len(template.targets) == 0:
				template.targets = self.targets

	@property
	def targets(self) -> list[str]:
		if self.__targets is None or len(self.__targets) == 0:
			self.__targets = set([])
			if self.templates is not None:
				for template in self.templates:
					if template.targets is not None:
						self.__targets.update(template.targets)
		# In the end, we return the overall list
		return list(self.__targets)

	@targets.setter
	def targets(self, targets: list[str]) -> None:
		self.__targets = set(targets)
		# Sets also the single templates' targets which are None
		for template in self.templates:
			if template.targets is None or len(template.targets) == 0:
				template.targets = self.__targets

	@property
	def genders(self) -> list[Gender]:
		return list(self.targets_by_gender.keys())


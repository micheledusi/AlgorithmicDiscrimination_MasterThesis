#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment analyzes the differences in gender prediction with a BERT model
# based on the job in the sentence.

import matplotlib
import numpy as np
from enum import IntEnum
from matplotlib import pyplot as plt
from transformers import pipeline
from src.parsers.occupations_parser import OccupationsParser
from settings import TOKEN_MASK
import settings

TOKEN_OCC = "[OCC]"


class Gender(IntEnum):
	NEUTER = 0
	MALE = 1
	FEMALE = 2

	@property
	def color(self) -> str:
		genders_colors: dict[Gender, str] = {
			Gender.NEUTER: "#E7C662",
			Gender.MALE: "#779be7",
			Gender.FEMALE: "#ef798a",
		}
		return genders_colors[self]


class Template:
	sentence: str
	targets: list[str] | None

	def __init__(self, sentence: str, targets: list[str] | None = None):
		self.sentence = sentence
		self.targets = targets


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


template_group_pronouns = TemplatesGroup("pronouns")
template_group_pronouns.templates = [
	Template(sentence=f"{TOKEN_MASK} worked as a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} studied for years to become a {TOKEN_OCC}."),
	Template(sentence=f"One day {TOKEN_MASK} will be a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} was a {TOKEN_OCC}.", targets=["i", "he", "she"]),
]
template_group_pronouns.targets = ["i", "you", "he", "she", "they"]
template_group_pronouns.targets_by_gender = {
	Gender.MALE: ["he"],
	Gender.FEMALE: ["she"],
}

template_group_personalnames = TemplatesGroup("personalnames")
template_group_personalnames.templates = [
	Template(sentence=f"{TOKEN_MASK} works as a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} is a {TOKEN_OCC}."),
	Template(sentence=f"{TOKEN_MASK} was a {TOKEN_OCC}."),
]
template_group_personalnames.targets = ["james", "robert", "john", "mary", "jennifer", "patricia"]
template_group_personalnames.targets_by_gender = {
	Gender.MALE: ["james", "robert", "john"],
	Gender.FEMALE: ["mary", "jennifer", "patricia"],
}

template_group_relatives = TemplatesGroup("relatives")
template_group_relatives.templates = [
	Template(sentence=f"My {TOKEN_MASK} works as a {TOKEN_OCC}."),
	Template(sentence=f"My {TOKEN_MASK} is a {TOKEN_OCC}."),
	Template(sentence=f"My {TOKEN_MASK} was a {TOKEN_OCC}."),
]
template_group_relatives.targets = ["mom", "mother", "dad", "father", "aunt", "uncle"]
template_group_relatives.targets_by_gender = {
	Gender.MALE: ["dad", "father"],
	Gender.FEMALE: ["mom", "mother"],
}


def extract_target_words(templates: dict[str, list[str]]) -> list[str]:
	"""
	Since every template in a templates group has its own targets, this method extract al the targets
	in the group and returns a list of them.
	:param templates: The templates group
	:return: The list of target words
	"""
	targets: set[str] = set()
	for _, tmpl_targets in templates.items():
		targets.update(tmpl_targets)
	print("Extracted targets: ", targets)
	return list(targets)


def compute_scores(templates_group: TemplatesGroup, occupations: list[str]) -> np.ndarray:
	"""
	Computes the scores of the "fill-mask" task for the BERT encoder.
	:param templates_group: The group of templates to analyze. It contains the list of templates to fill and the list
	of target words to use.
	:param occupations: The occupations to tune the templates.
	:return: A numpy array of shape: [# templates, # occupations, # target words]
	"""
	# Initializing the model
	unmasker = pipeline("fill-mask", model=settings.DEFAULT_BERT_MODEL_NAME,
	                    targets=templates_group.targets,
	                    top_k=len(templates_group.targets))
	# Initalizing the result
	scores: np.ndarray = np.zeros(
		shape=(len(templates_group.templates), len(occupations), len(templates_group.targets)))
	# For every template
	for i, tmpl in enumerate(templates_group.templates):
		print("Computing scores for template: ", tmpl.sentence)
		# For every occupation
		for j, occ in enumerate(occupations):
			tmpl_occ = tmpl.sentence.replace(TOKEN_OCC, occ)
			results = unmasker(tmpl_occ)

			results_aux: dict = {}
			for res in results:
				results_aux[res["token_str"]] = res["score"]
			# Saving the results for the current template and for the current occupation
			scores[i][j] = [results_aux[targ] for targ in templates_group.targets]
	return scores


def print_table_file(filepath: str, template: str, occupations: list[str], group_targets: list[str],
                     parser: OccupationsParser, data: np.ndarray) -> None:
	with open(filepath, 'w') as f:
		print(f'template', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'occupation', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		for tg in group_targets:
			print(f'{tg}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'stat_bergsma', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'stat_bls', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(file=f)

		for k, occ in enumerate(occupations):
			print(f'{template}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(f'{occ}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			for j in range(len(data[k])):
				print(f'{data[k, j]}', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(parser.get_percentage(occ, stat_name='bergsma'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(parser.get_percentage(occ, stat_name='bls'), end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
			print(file=f)
	return


def plot_image_bars_by_target(filepath: str, template: Template, group: TemplatesGroup, occupations: list[str],
                              data: np.ndarray) -> None:
	"""
	Plots a **Bars Graph** for the occupations in the list.
	:return: None
	"""
	tmpl_targets_ixs = [group.targets.index(t) for t in template.targets]

	occ_per_row = 15
	rows: int = int(np.ceil(len(occupations) / occ_per_row))
	fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(11, 9), dpi=150, sharex='all', sharey='all')

	bars_tot_width = 0.75
	bar_width = bars_tot_width / len(tmpl_targets_ixs)
	x = np.arange(occ_per_row)

	for curr_row, ax in enumerate(axs):
		# Number of occupations in this subplot (figure row)
		occ_in_curr_row = occ_per_row if occ_per_row * (curr_row + 1) <= len(occupations) \
			else len(occupations) % occ_per_row

		# Starting index of the occupations
		occ_start_ix = occ_per_row * curr_row
		# Indices slice for the occupations
		subplot_occs_ixs = slice(occ_start_ix, occ_start_ix + occ_in_curr_row)

		# Data for the current subplot
		subplot_data = data[subplot_occs_ixs]
		# Subplot data dimensions: [# subplot_occupations, # targets]
		# print("subplot_data.shape: ", subplot_data.shape)

		subplot_x = x[:occ_in_curr_row]
		# print("subplot_x.shape: ", subplot_x.shape)

		cmap = matplotlib.cm.get_cmap('Set2')

		for j_local_ix, j in enumerate(tmpl_targets_ixs):
			subplot_row_data = subplot_data[..., j]
			# print("subplot_row_data.shape: ", subplot_row_data.shape)
			target_subplot_x = subplot_x + (bar_width * j_local_ix + 1.0 - bars_tot_width)
			# print("target_subplot_x.shape: ", target_subplot_x.shape)
			ax.bar(target_subplot_x, subplot_row_data, bar_width, label=group.targets[j], zorder=5, color=cmap(j))

		for k, occ_label in enumerate(occupations[subplot_occs_ixs]):
			ax.annotate(occ_label, xy=(k, 0.0), xytext=(-5, 10), textcoords="offset points",
			            rotation=90, fontsize=10, zorder=10)
		ax.set_ylabel('Scores')
		ax.tick_params(bottom=True, labelbottom=False)
		ax.set_xticks(subplot_x)
		ax.grid(visible=True, axis='y', zorder=0)

	axs[0].legend(bbox_to_anchor=(0.0, 1.2, 1.0, 0.102), loc='upper center', ncol=len(template.targets),
	              mode="", borderaxespad=0.)
	fig.suptitle(template.sentence)
	plt.savefig(filepath)
	# plt.show()
	return


def plot_image_bars_by_gender(filepath: str, template: Template, group: TemplatesGroup,
                              occupations: list[str], data: np.ndarray) -> None:
	"""
	Plots a **Bars Graph** for the occupations in the list.
	:return: None
	"""
	# Current data dimensions: [# occupations, # targets]
	# We want to reduce it to [# occupations, # genders] by merging slices on axis=1
	data_by_gender = np.zeros(shape=(len(occupations), len(group.targets_by_gender)))

	# First, we extract the indices for each gender
	for gender_ix, (_, gender_targets) in enumerate(group.targets_by_gender.items()):
		current_gender_indices = [group.targets.index(t) for t in gender_targets]
		current_gender_data = np.mean(data[..., current_gender_indices], axis=1)
		data_by_gender[:, gender_ix] = current_gender_data

	# Now, we sort data_by_gender by the first gender in descending order
	sorting_ixs = (-data_by_gender[:, 0]).argsort()
	# We now re-create 'data_by_gender' and 'occupations'
	data_by_gender = data_by_gender[sorting_ixs, :]
	occupations = [occupations[i] for i in sorting_ixs]

	# Defining number of occupations per row
	occ_per_row = 30
	rows: int = int(np.ceil(len(occupations) / occ_per_row))
	# print("Number of rows: ", rows)
	fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(13, 8), dpi=150, sharey='all')

	bars_tot_width = 0.6
	bar_width = bars_tot_width / len(group.targets_by_gender)
	x = np.arange(occ_per_row)

	for curr_row, ax in enumerate(axs):
		# print("\tCurrent row: ", curr_row)

		# Number of occupations in this subplot (figure row)
		occ_in_curr_row = occ_per_row if occ_per_row * (curr_row + 1) <= len(occupations) \
			else len(occupations) % occ_per_row

		# Starting index of the occupations
		occ_start_ix = occ_per_row * curr_row
		# Indices slice for the occupations
		subplot_occs_ixs = slice(occ_start_ix, occ_start_ix + occ_in_curr_row)

		# Data for the current subplot
		subplot_data = data_by_gender[subplot_occs_ixs]
		# Subplot data dimensions: [# subplot_occupations, # genders]

		subplot_x = x[:occ_in_curr_row]
		# print("subplot_x.shape: ", subplot_x.shape)

		for j, gender in enumerate(group.targets_by_gender):
			subplot_row_data = subplot_data[..., j]
			# print("subplot_row_data.shape: ", subplot_row_data.shape)
			target_subplot_x = subplot_x + (bar_width * j) - (bars_tot_width / 2)
			# print("target_subplot_x.shape: ", target_subplot_x.shape)
			label: str = f"{gender.name.lower()}: {group.targets_by_gender[gender]}"
			ax.bar(target_subplot_x, subplot_row_data, bar_width, label=label, zorder=5, color=gender.color)

		ax.set_ylabel('Scores')
		ax.tick_params(bottom=True, labelbottom=True)
		ax.set_xticks(subplot_x, occupations[subplot_occs_ixs], rotation=90)
		ax.grid(visible=True, axis='y', zorder=0)

	axs[0].legend(bbox_to_anchor=(0.0, 1.2, 1.0, 0.05), loc='upper center', ncol=len(group.targets_by_gender),
	              mode="", borderaxespad=0.)
	fig.suptitle(template.sentence)
	fig.tight_layout()
	plt.savefig(filepath)
	# plt.show()
	return


def launch() -> None:
	# Extracting the list of occupations from WinoGender dataset
	parser = OccupationsParser()
	occs_list: list[str] = parser.occupations_list

	groups = [
		template_group_pronouns,
		template_group_personalnames,
		template_group_relatives,
	]

	for g_ix, group in enumerate(groups):
		# Computing scores
		scores: np.ndarray = compute_scores(templates_group=group, occupations=occs_list)

		for i, tmpl in enumerate(group.templates):
			data = scores[i]
			# Data dimensions: [# occupations, # targets]

			tmpl_targets = tmpl.targets

			# Plotting the bar scores graph for each template
			plot_image_bars_by_target(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/img/'
				         f'group_{group.name}_by_targets_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=data,
			)

			# Plotting the bar scores graph for each template
			plot_image_bars_by_gender(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/img/'
				         f'group_{group.name}_by_genders_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				group=group,
				occupations=occs_list,
				data=data,
			)

			# Printing one table for each template
			print_table_file(
				filepath=f'{settings.FOLDER_RESULTS}/gender_prediction/tables/'
				         f'group_{group.name}_by_targets_{i:02d}.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
				template=tmpl.sentence,
				occupations=occs_list,
				group_targets=group.targets,
				parser=parser,
				data=data,
			)

	return

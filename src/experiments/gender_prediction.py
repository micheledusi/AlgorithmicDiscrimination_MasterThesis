#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This experiment analyzes the differences in gender prediction with a BERT model
# based on the job in the sentence.
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from transformers import pipeline
from src.parsers.occupations_parser import OccupationsParser
from settings import TOKEN_MASK
import settings

TOKEN_OCC = "[OCC]"

pronouns_templates = {
	f"{TOKEN_MASK} worked as a {TOKEN_OCC}.": ["i", "you", "he", "she", "they"],
	f"{TOKEN_MASK} studied for years to become a {TOKEN_OCC}.": ["i", "you", "he", "she", "they"],
	f"One day {TOKEN_MASK} will be a {TOKEN_OCC}.": ["i", "you", "he", "she", "they"],
	f"{TOKEN_MASK} was a {TOKEN_OCC}.": ["i", "he", "she"],
}
# The three most frequent male and three most frequent female english names
gendered_names_templates = {
	f"{TOKEN_MASK} works as a {TOKEN_OCC}.": ["james", "robert", "john", "mary", "jennifer", "patricia"],
	f"{TOKEN_MASK} is a {TOKEN_OCC}.": ["james", "robert", "john", "mary", "jennifer", "patricia"],
}
# Three male and three females relatives nouns
relatives_templates = {
	f"My {TOKEN_MASK} works as a {TOKEN_OCC}.": ["mom", "mother", "dad", "father", "aunt", "uncle"],
	f"My {TOKEN_MASK} is a {TOKEN_OCC}.": ["mom", "mother", "dad", "father", "aunt", "uncle"],
	f"My {TOKEN_MASK} was a {TOKEN_OCC}.": ["mom", "mother", "dad", "father", "aunt", "uncle"],
}

templates_groups = [
	pronouns_templates,
	gendered_names_templates,
	relatives_templates,
]


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


def compute_scores(templates: dict[str, list[str]], occupations: list[str], target_words: list[str]) -> np.ndarray:
	"""
	Computes the scores of the "fill-mask" task for the BERT encoder.
	:param templates: The dictionary of templates to fill.
	:param occupations: The occupations to tune the templates.
	:param target_words: The words to score.
	:return: A numpy array of shape: [# templates, # occupations, # targets]
	"""
	# Initializing the model
	unmasker = pipeline("fill-mask", model=settings.DEFAULT_BERT_MODEL_NAME, targets=target_words, top_k=len(target_words))
	# Initalizing the result
	scores: np.ndarray = np.zeros(shape=(len(templates), len(occupations), len(target_words)))
	# For every template
	for i, tmpl in enumerate(templates):
		print(tmpl)
		# For every occupation
		for j, occ in enumerate(occupations):
			tmpl_occ = tmpl.replace(TOKEN_OCC, occ)
			results = unmasker(tmpl_occ)

			results_aux: dict = {}
			for res in results:
				results_aux[res["token_str"]] = res["score"]
			# Saving the results for the current template and for the current occupation
			scores[i][j] = [results_aux[targ] for targ in target_words]
	return scores


def print_table_file(filepath: str, template: str, occupations: list[str], targets: list[str],
                     parser: OccupationsParser, data: np.ndarray) -> None:
	with open(filepath, 'w') as f:
		print(f'template', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		print(f'occupation', end=settings.OUTPUT_TABLE_COL_SEPARATOR, file=f)
		for tg in targets:
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


def plot_bars_image(filepath: str, template: str, occupations: list[str],
                    group_targets: list[str], tmpl_targets: list[str],
                    data: np.ndarray) -> None:
	"""
	Plots a **Bars Graph** for the occupations in the list.
	:return: None
	"""
	tmpl_targets_ixs = [group_targets.index(t) for t in tmpl_targets]

	occ_per_row = 16
	rows: int = int(np.ceil(len(occupations) / occ_per_row))
	fig, axs = plt.subplots(nrows=rows, ncols=1, figsize=(10, 7), dpi=150, sharex='all')

	width = 0.8
	sub_width = width / len(tmpl_targets_ixs)
	x = np.arange(occ_per_row)

	for curr_row, ax in enumerate(axs):
		# Number of occupations in this subplot (figure row)
		occ_in_curr_row = occ_per_row if occ_per_row * (curr_row + 1) < len(occupations) else len(occupations) % occ_per_row
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

		for j in tmpl_targets_ixs:
			subplot_row_data = subplot_data[..., j]
			# print("subplot_row_data.shape: ", subplot_row_data.shape)
			target_subplot_x = subplot_x + (sub_width * j) - width / 2
			# print("target_subplot_x.shape: ", target_subplot_x.shape)
			ax.bar(target_subplot_x, subplot_row_data, sub_width, label=group_targets[j], zorder=5, color=cmap(j * 0.125))

		for k, occ_label in enumerate(occupations[subplot_occs_ixs]):
			ax.annotate(occ_label, xy=(k - 0.5, 0.0), xytext=(0, 10), textcoords="offset points",
			            rotation=90, fontsize=10, zorder=10)
		ax.set_ylabel('Scores')
		ax.tick_params(bottom=False, labelbottom=False)
		ax.grid(visible=True, axis='y', zorder=0)

	axs[0].legend(bbox_to_anchor=(0.0, 1.05, 1.0, 0.102), loc='upper center', ncol=len(tmpl_targets),
	              mode="", borderaxespad=0.)
	fig.suptitle(template)
	# fig.tight_layout()
	plt.savefig(filepath)
	# plt.show()
	return


def launch() -> None:
	# Extracting the list of occupations from WinoGender dataset
	parser = OccupationsParser()
	occs_list: list[str] = parser.occupations_list

	for g, group in enumerate(templates_groups):

		# Selecting the current templates group
		templates: dict[str, list[str]] = templates_groups[g]

		# Computing scores
		targets: list[str] = extract_target_words(templates)
		scores: np.ndarray = compute_scores(templates=templates, occupations=occs_list, target_words=targets)
		# print("Scores shape: ", scores.shape)

		for i, tmpl in enumerate(templates):
			data = scores[i]
			# Data dimensions: [# occupations, # targets]

			tmpl_targets = templates[tmpl]

			# Plotting the bar scores graph for each template
			plot_bars_image(
				filepath=f'results/gender_prediction/img/group_{g:02d}_tmpl_{i:02d}.{settings.OUTPUT_IMAGE_FILE_EXTENSION}',
				template=tmpl,
				occupations=occs_list,
				group_targets=targets,
				tmpl_targets=tmpl_targets,
				data=data,
			)

			# Printing one table for each template
			print_table_file(
				filepath=f'results/gender_prediction/tables/group_{g:02d}_tmpl_{i:02d}.{settings.OUTPUT_TABLE_FILE_EXTENSION}',
				template=tmpl,
				occupations=occs_list,
				targets=targets,
				parser=parser,
				data=data,
			)

	return

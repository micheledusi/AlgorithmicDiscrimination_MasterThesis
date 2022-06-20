#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# Plotting functions for prediction task

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from src.models.templates import Template, TemplatesGroup


def plot_image_bars_by_target(filepath: str, template: Template, group: TemplatesGroup,
                              occupations: list[str], data: np.ndarray) -> None:
	"""
	Plots and saves an image graph representing the prediction values from the gender prediction task.
	The bars will be indicated by occupations. There will be a set of bars for each target word.
	:param filepath: The filepath where to save the image.
	:param template: The template to plot
	:param group: The group of templates it belongs to
	:param occupations: The list of occupations to plot
	:param data: The result data to visualize
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

		cmap = cm.get_cmap('Set2')

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
	Plots and saves an image graph representing the prediction values from the gender prediction task.
	The bars will be indicated by occupations. There will be a set of bars for each gender in the group.
	:param filepath: The filepath where to save the image.
	:param template: The template to plot
	:param group: The group of templates it belongs to
	:param occupations: The list of occupations to plot
	:param data: The result data to visualize
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

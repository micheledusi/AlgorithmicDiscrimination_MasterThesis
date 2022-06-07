#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This class aims to visualize tokens surprise along the 13 layers of BERT for a pair of gender-opposite sentences.

from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt


class PairSurpriseHeatmapsPlotter:
	"""
	This class stores and visualizes heatmaps for surprise scores.
	"""
	__scores: tuple = None
	__labels: tuple[list[str], list[str]] = None
	__difference = None
	__ready_to_show = False

	def __init__(self, pair_result: dict):
		self.set_pair_result(pair_result)

	def set_pair_result(self, pair_result: dict) -> None:
		"""
		Sets the attributes of this object by extracting them out of a result produced by an AnomalyModel.
		:param pair_result: a dictionary with tokens, scores and scores difference.
		:return: None
		"""
		self.scores = pair_result["scores"]
		self.labels = pair_result["tokens"]
		self.__difference = pair_result["difference"]
		# Since it's reset
		self.__ready_to_show = False

	@property
	def scores(self) -> tuple:
		return self.__scores

	@scores.setter
	def scores(self, scores: tuple) -> None:
		self.__scores = scores
		return

	@property
	def labels(self) -> tuple[list[str], list[str]]:
		return self.__labels

	@labels.setter
	def labels(self, labels: tuple[list[str], list[str]]) -> None:
		if labels is None:
			# The <None> value is allowed
			self.__labels = None
		self.__labels = labels

	def plot_surprise_heatmaps(self) -> None:
		"""
		Creates the plot for the two heatmaps and the difference heatmap.
		This can be saved in a figure or shown with the methods "show" or "save".
		:return: None
		"""
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey='all', dpi=150)

		# Splitting scores and labels
		scores_l, scores_r = self.scores
		labels_l, labels_r = self.labels

		num_layers, num_tokens = scores_l.shape
		fig.suptitle(f'Surprise heatmaps (layers={num_layers}, tokens={num_tokens})')

		ax1.imshow(scores_l, aspect="auto")
		ax1.set_title("Male sentence")
		ax1.set_xticks(range(len(labels_l)), labels_l, rotation=90, fontsize=8)
		ax1.set_box_aspect(2)

		ax3.imshow(scores_r, aspect="auto")
		ax3.set_title("Female sentence")
		ax3.set_xticks(range(len(labels_r)), labels_r, rotation=90, fontsize=8)
		ax3.set_box_aspect(2)

		ax2.imshow(self.__difference, aspect="auto", cmap="RdPu")
		ax2.set_title("Absolute difference")
		ax2.set_xticks(range(len(labels_r)), labels_r, rotation=90, fontsize=8)
		ax2.set_box_aspect(2)

		self.__ready_to_show = True
		return

	def show(self):
		"""
		Wrapper for the pyplot function "show".
		This works only if the user previously built the plot correctly (i.e. by calling a "plot_something" method)
		:return: The value returned by pyplot.show()
		"""
		if not self.__ready_to_show:
			raise RuntimeWarning("Cannot show non-existent plot")
		else:
			return plt.show()

	def save(self, filename: str, timestamp: bool = False):
		"""
		Saves the plot as an image.
		:param filename: The path and name of the file, with the desired extension (.png / .jpg / .pdf)
		:param timestamp: If True, adds a timestamp to the file name. Thus, OVERWRITING new versions of the same
			file will not be possible anymore: every file will be uniquely saved (assuming no files are saved within a
			period less than 1 second).
		:return: the value returned by <savefig>.
		"""
		if not self.__ready_to_show:
			raise RuntimeWarning("Cannot save non-existent plot")
		if timestamp:
			tstamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			# Inserting the timestamp between the filename and the extension
			path = Path(filename)
			filename = path.parent / f"{path.stem}_{tstamp}{path.suffix}"
		return plt.savefig(fname=filename)



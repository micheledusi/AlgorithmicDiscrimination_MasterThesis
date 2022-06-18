#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This script contains functions used to plot word embeddings.
# It uses PyTorch Tensors as data, and PyPlot to visualize them.
# To use this module, instance an object of class "EmbeddingsPlotter".

import typing
from datetime import datetime
from pathlib import Path
from torch import Tensor, pca_lowrank, squeeze
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import settings


class EmbeddingsScatterPlotter:
	"""
	This class stores and visualizes embeddings.
	It can operate dimensionality reduction to produce 2D and 3D graphs.
	"""

	DEFAULT_COLORMAP: str = settings.GENDER_CYAN2PINK_COLORMAP_NAME

	__embeddings = None
	__pca_2d_vectors: Tensor = None
	__pca_3d_vectors: Tensor = None
	__labels: [str] = None
	__colors: [float] = None
	__sizes: [float] = None
	__colormap: str = DEFAULT_COLORMAP
	__cmap_norm: Normalize = None
	__ready_to_show: bool = False
	__figure = None

	def __init__(self, embeddings: Tensor = None):
		self.embeddings = embeddings
		self.__reset_derived_attributes()
		return

	@property
	def embeddings(self) -> Tensor:
		return self.__embeddings

	@embeddings.setter
	def embeddings(self, embeddings: Tensor) -> None:
		# If the layer dimension (dim_ix = 1) is futile, i.e. there's a single layer, that dimension is squeezed
		if len(embeddings.size()) == 3 and embeddings.size()[1] == 1:
			embeddings = squeeze(embeddings, dim=1)
		self.__embeddings = embeddings
		self.__reset_derived_attributes()
		return

	@property
	def labels(self) -> [str]:
		return self.__labels

	@labels.setter
	def labels(self, labels: [str]) -> None:
		if labels is None:
			# The <None> value is allowed
			self.__labels = None
		elif self.count() != len(labels):
			# The number of labels MUST be equals to the number of embeddings
			raise RuntimeError("Labels number is different from embeddings count")
		self.__labels = labels

	@property
	def colors(self) -> [float]:
		return self.__colors

	@colors.setter
	def colors(self, colors: [float]) -> None:
		if colors is None:
			# The <None> value is allowed
			self.__colors = None
		elif self.count() != len(colors):
			# The number of colors MUST be equals to the number of embeddings
			raise RuntimeError("Labels number is different from embeddings count")
		else:
			self.__colors = colors
			# Setting the min and max value for the colormap
			self.__cmap_norm = Normalize(vmin=min(self.colors), vmax=max(self.colors))

	@property
	def sizes(self) -> [float]:
		return self.__sizes

	@sizes.setter
	def sizes(self, sizes: [float]) -> None:
		if sizes is None:
			# The <None> value is allowed
			self.__sizes = None
		elif self.count() != len(sizes):
			# The number of sizes MUST be equals to the number of embeddings
			raise RuntimeError("Labels number is different from embeddings count")
		self.__sizes = sizes

	@property
	def colormap(self) -> str:
		return self.__colormap

	@colormap.setter
	def colormap(self, colormap: str) -> None:
		plt.set_cmap(colormap)
		self.__colormap = colormap

	def count(self) -> int:
		"""
		Returns the number of embeddings saved into this plotter.
		:return: The count of embeddings.
		"""
		return self.embeddings.size()[0]

	def __reset_derived_attributes(self) -> None:
		self.__pca_2d_vectors = None
		self.__pca_3d_vectors = None
		self.labels = None
		self.colors = None
		self.sizes = None
		self.colormap = self.DEFAULT_COLORMAP
		self.__cmap_norm = None
		self.__ready_to_show = False
		self.__figure = None

	def compute_pca_2d_vectors(self) -> Tensor:
		"""
		Computes the two-dimensional principal components with the PCA (Principal Components Analysis).
		The embeddings must be set correctly with a tensor of size (# samples, # features).
		:return: The computed tensor of size (# samples, 2)
		"""
		if self.embeddings is None:
			# If the embeddings are empty
			raise RuntimeError("Empty embeddings: cannot compute 2D PCA")
		self.__pca_2d_vectors, S, V = pca_lowrank(self.embeddings, q=2)
		return self.__pca_2d_vectors

	def compute_pca_3d_vectors(self) -> Tensor:
		"""
		Computes the three-dimensional principal components with the PCA (Principal Components Analysis).
		The embeddings must be set correctly with a tensor of size (# samples, # features).
		:return: The computed tensor of size (# samples, 3)
		"""
		if self.embeddings is None:
			# If the embeddings are empty
			raise RuntimeError("Empty embeddings: cannot compute 3D PCA")
		self.__pca_3d_vectors, S, V = pca_lowrank(self.embeddings, q=3)
		return self.__pca_3d_vectors

	def plot_2d_pc(self) -> None:
		"""
		Creates a 2D plot with the principal components of the embeddings, using the procedure of PCA.
		If the embeddings have multiple points, they're connected with lines showing how the embeddings evolve through the layers.
		:return: None
		"""
		# If the 2D coordinates are not computed yet:
		if self.__pca_2d_vectors is None:
			self.compute_pca_2d_vectors()
		# We extract the Xs and Ys of the points
		coords = self.__pca_2d_vectors.moveaxis(-1, 0)  # Brings the last dimension (the one reduced with PCA) to front
		coords = coords.detach().numpy()  # Converting into NumPy array
		xs, ys = coords[0], coords[1]

		self.__figure, ax = plt.subplots()

		if len(xs.shape) == 1 and len(ys.shape) == 1:
			# If there is no history, points are individually visualized with no connections between them
			ax.scatter(xs, ys, c=self.colors, s=self.sizes)

			# If labels are set, the plot is annotated by points
			if self.labels is not None:
				for label, x, y in zip(self.labels, xs, ys):
					ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points", fontsize='small')

		elif len(xs.shape) == 2 and len(ys.shape) == 2:
			# Every embedding is a history of N points, where N = #layers of encoder (e.g. BERT)
			# For BERT, we'll have 13 points for each embedding
			for i, (xh, yh, col_float) in enumerate(zip(xs, ys, self.colors)):
				# xh and yh represent the history of each embedding across the layers
				layers = len(xh)
				normalized_col_float = self.__cmap_norm(col_float)
				color = plt.get_cmap(self.colormap)(normalized_col_float)
				ax.plot(xh, yh, color="#aaaaaa", alpha=0.1)
				ax.scatter(xh, yh, color=color, s=range(layers))

				if self.labels is not None:
					label = self.labels[i]
					ax.annotate(label, xy=(xh[-1], yh[-1]), xytext=(1, 1), textcoords="offset points")

		# Allowing the visualization
		self.__ready_to_show = True
		return

	def plot_3d_pc(self) -> None:
		"""
		Creates a 3D plot with the principal components of the embeddings, using the procedure of PCA.
		:return: None
		"""
		# If the 3D coordinates are not computed yet:
		if self.__pca_3d_vectors is None:
			self.compute_pca_3d_vectors()
		# We extract the Xs, Ys and Zs of the points
		coords = self.__pca_2d_vectors.moveaxis(-1, 0)  # Brings the last dimension (the one reduced with PCA) to front
		coords = coords.detach().numpy()
		x, y, z = coords[0], coords[1], coords[2]

		self.__figure = plt.figure()
		ax = self.__figure.add_subplot(projection='3d')
		ax.scatter(x, y, z, c=self.colors, s=self.sizes)

		# If labels are set, the plot is annotated by points
		if self.labels is not None:
			for label, x, y, z in zip(self.labels, x, y, z):
				ax.text(x, y, z, label)
		# Allowing the visualization
		self.__ready_to_show = True
		return

	def show(self) -> typing.Any:
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

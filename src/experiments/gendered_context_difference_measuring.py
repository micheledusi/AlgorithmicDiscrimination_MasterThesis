#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################
from matplotlib.colors import Normalize

from src.models.word_encoder import WordEncoder
from src.parsers.occupations_parser import OccupationsParser
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter
import torch
import matplotlib.pyplot as plt


def launch() -> None:
	enc_m, enc_f = WordEncoder(), WordEncoder()
	enc_m.set_embedding_template("[CLS] he is a %s [SEP]", 4)
	enc_f.set_embedding_template("[CLS] she is a %s [SEP]", 4)

	# Extracting the list of occupations sorted by the highest female percentage, from WinoGender dataset
	parser = OccupationsParser()
	occs_number = 20
	occs_list = parser.get_sorted_female_occupations(max_length=occs_number, stat_name="bergsma", female_percentage="highest")

	selected_layers = list(range(7, 13))
	emb_list_m = []
	emb_list_f = []
	occ_list_m = []
	occ_list_f = []
	measures = {}
	similarity_fun = torch.nn.CosineSimilarity(dim=1)

	# For every occupation
	for occ, pct in occs_list:
		# <occ> is the occupation string
		# <pct> is the percentage float indicating the female presence in the real-world occupation
		print(f"\tStudying embeddings for word <{occ}> in layers {selected_layers}")

		emb_m = enc_m.embed_word(occ, layers=selected_layers)
		emb_f = enc_f.embed_word(occ, layers=selected_layers)
		emb_list_m.append(emb_m)
		emb_list_f.append(emb_f)

		occ_list_m.append(occ + "_m")
		occ_list_f.append(occ + "_f")

		# Computing metrics
		male_pct: float = 100.0 - pct
		pct_dist: float = abs(pct - male_pct)
		emb_simi = similarity_fun(emb_m, emb_f)
		# Adding results to the measures dictionary
		measures[occ] = (pct_dist, emb_simi)

	# Turning embeddings lists into PyTorch tensors
	embeddings = torch.stack([*emb_list_m, *emb_list_f])
	print("Embeddings size: ", embeddings.size())

	# Visualizing embeddings
	plotter = EmbeddingsScatterPlotter(embeddings)
	plotter.colors = [*([0] * occs_number), *([1] * occs_number)]
	plotter.labels = [*occ_list_m, *occ_list_f]
	plotter.plot_2d_pc()
	plotter.show()

	# Visualizing measures
	plt.figure()
	cmap = plt.get_cmap("viridis")
	norm = Normalize(vmin=0, vmax=100)
	for occ, meas in measures.items():
		pct_dist: float = meas[0]
		emb_simi = meas[1].detach().numpy()
		pct_color = cmap(norm(pct_dist))
		plt.plot(selected_layers, emb_simi, '.-', color=pct_color, label=occ)

		last_point = (selected_layers[-1], emb_simi[-1])
		plt.annotate(occ, xy=last_point, xytext=(5, -2), textcoords="offset points")

	plt.xlabel("layers")
	plt.ylabel("cosine similarity")
	plt.title("Similarity between occupations in gender-opposite contexts")
	plt.show()
	return



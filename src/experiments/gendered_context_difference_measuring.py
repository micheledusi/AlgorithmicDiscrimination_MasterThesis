#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

from src.models.word_encoder import WordEncoder
from src.parsers.occupations_parser import OccupationsParser
from src.viewers.plot_scatter_embeddings import EmbeddingsScatterPlotter
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def launch() -> None:
	enc_m, enc_f = WordEncoder(), WordEncoder()
	enc_m.set_embedding_template("[CLS] he worked as a %s [SEP]", 5)
	enc_f.set_embedding_template("[CLS] she worked as a %s [SEP]", 5)

	# enc_n.set_embedding_template("[CLS] %s [SEP]", 1)
	# enc_n.set_embedding_template("[CLS] I work as %s [SEP]", 4)

	# Extracting the list of occupations sorted by the highest female percentage, from WinoGender dataset
	parser = OccupationsParser()
	occs_number = 60
	occs_list = [*parser.get_sorted_female_occupations(max_length=(occs_number // 2), stat_name="bls", female_percentage="highest"),
	             *parser.get_sorted_female_occupations(max_length=(occs_number // 2), stat_name="bls", female_percentage="lowest")]

	selected_layers = range(0, 13)
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
		# print(f"\tStudying embeddings for word <{occ}> in layers {selected_layers}")

		emb_m = enc_m.embed_word_merged(occ, layers=selected_layers)
		emb_f = enc_f.embed_word_merged(occ, layers=selected_layers)
		emb_list_m.append(emb_m)
		emb_list_f.append(emb_f)

		occ_list_m.append(occ + "_m")
		occ_list_f.append(occ + "_f")

		# Computing metrics
		male_pct: float = 100.0 - pct
		pct_dist: float = abs(pct - male_pct)
		emb_simi = similarity_fun(emb_m, emb_f)
		emb_dist = (emb_m - emb_f).pow(2).sum(1).sqrt()     # Computes euclidean distance by layers
		# Adding results to the measures dictionary
		measures[occ] = (pct_dist, emb_simi, emb_dist)

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

	pearson_x, pearson_y = [], []
	for occ, meas in measures.items():
		pct_dist: float = meas[0]
		emb_simi = meas[1].detach().numpy()
		emb_dist = meas[2].detach().numpy()
		plotted_measure = emb_simi

		pct_color = cmap(norm(pct_dist))
		plt.plot(selected_layers, plotted_measure, '.-', color=pct_color, label=occ)

		last_point = (selected_layers[-1], plotted_measure[-1])
		plt.annotate(occ, xy=last_point, xytext=(5, -2), textcoords="offset points")

		pearson_x.append(pct_dist)
		pearson_y.append(plotted_measure[-1])

	plt.xlabel("layers")
	plt.ylabel("cosine similarity")
	plt.title("Similarity between occupations in gender-opposite contexts")
	plt.show()

	# Correlation
	corr_tensor = torch.stack([torch.Tensor(pearson_x), torch.Tensor(pearson_y)])
	corr = torch.corrcoef(corr_tensor)[0][1]
	print("Correlation coefficient between real word disparity and measured cosine similarity: ", corr)

	return



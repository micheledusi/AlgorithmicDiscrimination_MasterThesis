#########################################################################
#                            Dusi's Thesis                              #
# Algorithmic Discrimination and Natural Language Processing Techniques #
#########################################################################

# This model tries to define a gender subspace in the embeddings space, for each layers.
# For the sake of simplicity and analysis, we'll consider only male and female genders, which
# is not a socially correct assumption.

from abc import abstractmethod, ABC
from enum import IntEnum

import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


class _AbstractGenderClassifier(ABC):
	"""
	A generic classifier for embeddings, that detects gender (or, if desired, other classes).
	Cannot be instanced directly, but requires a subclass to work properly.
	The subclass must specify the type of classifier that work on each level.
	"""

	def __init__(self, name: str, training_embeddings: np.ndarray, training_genders: list[IntEnum] | list[int],
	             layers_labels: list[str] = None, print_summary: bool = False) -> None:
		"""
		Creates an embedding classifier that detects the gender.

		:param name: The name of the model.
		:param training_embeddings: The training data, of dimensions: [# samples, # layers, # features]
		:param training_genders: The training "labels", i.e. the genders of embeddings [# samples, # layers]
		:param layers_labels: The list of names for the considered layers, if these do not correspond to the standard
		indices from 0 to N-1.
		:param print_summary: If True, prints a brief summary of dimensions.
		"""
		self.name = name
		self.__num_features: int
		num_samples, num_layers, self.__num_features = training_embeddings.shape
		assert num_samples == len(training_genders)

		if print_summary:
			print("New gender model with inner classifiers: ", type(self._instance_new_classifier()))
			print("Trained on samples: ", num_samples)
			print("Number of layers: ", num_layers)
			print("Number of features: ", self.__num_features)

		# Training a classifier for each layer
		self._classifiers: list = []
		for layer in range(num_layers):
			clf = self._instance_new_classifier()
			train_x = training_embeddings[:, layer]
			clf.fit(train_x, training_genders)
			self._classifiers.append(clf)
		assert self.num_layers == num_layers
		# Layers labels
		self.__layers_labels = layers_labels
		if self.__layers_labels is not None:
			assert len(self.__layers_labels) == self.num_layers

	def __str__(self) -> str:
		return f"Model '{self.name}' with {self.num_layers} classifiers of type {type(self._instance_new_classifier())}"

	@property
	def classifiers(self) -> list:
		return self._classifiers.copy()

	@property
	def num_layers(self) -> int:
		return len(self._classifiers)

	@abstractmethod
	def _instance_new_classifier(self):
		raise NotImplementedError("Cannot instance directly a '_GenderClassifier' object. Please use a child class.")

	@property
	@abstractmethod
	def features_importance(self) -> np.ndarray:
		"""
		Returns an array of dimensions [# layers, # features].
		Each value represents the "importance" of the feature af the layer; this "importance" measures how relevant
		a feature is in the computation of the gender.

		Please, note that this method may return different "kinds" of array according to the inner classifiers.

		:return: An array representing the relevance of each feature of each layer.
		"""
		pass

	@property
	@abstractmethod
	def features_bias(self) -> np.ndarray:
		"""
		Returns an array of dimensions [# layers, # features].
		Each value represents the feature contribution towards the male of female gender class during the classification.
		In other words, this could be interpreted as the gender "direction" in the features space.

		:return: An array representing the gender bias of each feature of each layer.
		"""
		pass

	def get_most_important_features(self, cut_zeros: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
		"""
		Returns the most important features, with the corresponding importance measure.

		The result is a list where each element corresponds to a layer.
		The layer-associated element is a tuple composed by two arrays:
			- The array of indices of the most significant features of that layer.
			- The array of values associated to those indices.

		Each one of the two arrays is a 1D array with dimensions: [# <= features]

		If the given parameter is true, the zero-importance features are removed from the result; then the result for
		that layer may have a dimension inferior to the original number of features.

		:param cut_zeros: If True, removes the zero-importance features from the arrays.
		:return: A list of couples representing the indices and the values of the most useful features in the classifier.
		"""
		result: list[tuple[np.ndarray, np.ndarray]] = []
		for layer in range(self.num_layers):
			sorted_importance_indices = np.argsort(-self.features_importance[layer], axis=-1)
			sorted_importance = self.features_importance[layer, sorted_importance_indices]
			if cut_zeros:
				# Finding the non-zero elements
				non_zeros_indices = np.argwhere(sorted_importance)
				# Extracting the non-zero elements
				sorted_importance_indices = sorted_importance_indices[non_zeros_indices]
				sorted_importance = sorted_importance[non_zeros_indices]
			result.append((
				np.squeeze(sorted_importance_indices),
				np.squeeze(sorted_importance)
			))
		return result

	def _check_classifiers_method(self, method_name: str, raise_exception: bool = True) -> bool:
		"""
		Checks if all the inner classifiers have a given method.
		If not, by default it raises an exception. Otherwise, you can tune the second parameter to return a boolean.

		:param method_name: The method you want to search.
		:param raise_exception: If True, raises an exception if the method is not found in anyone of the classifiers.
		Otherwise, the method returns a boolean value.
		:return: True if all the inner classifiers have the given method.
		"""
		for label, clf in zip(self.__layers_labels, self._classifiers):
			attr = getattr(clf, method_name, None)
			if attr is None or not callable(attr):
				if raise_exception:
					raise AttributeError(f"Cannot call method {method_name} on classifier of layer '{label}'.")
				else:
					return False
		return True

	def score_accuracy(self, embeddings: np.ndarray, genders: np.ndarray) -> np.ndarray:
		"""
		This method uses the "score" function of the classifiers to compute accuracy for each layer.
		If the used classifier has no "score" method, this function raises an exception.

		:param embeddings: The embeddings to use to evaluate
		:param genders: The list of correct gender labels
		:return: A numpy array of accuracies, one for each layer
		"""
		self._check_classifiers_method("score")
		accuracies = np.zeros(shape=self.num_layers)
		for layer in range(self.num_layers):
			clf = self._classifiers[layer]
			accuracies[layer] = clf.score(embeddings[:, layer], genders)
		return accuracies

	def evaluate(self, evaluation_embeddings: list[np.ndarray], evaluation_genders: list[IntEnum] | list[int]):
		accuracies = self.score_accuracy(np.asarray(evaluation_embeddings), np.asarray(evaluation_genders))
		for label, acc in zip(self.__layers_labels, accuracies):
			print(f"Layer {label:s}: acc = {acc:6.4%}")

	def predict_gender_class(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method predicts the gender of the embeddings, according to the trained inner classifiers.
		Each embedding is considered as a group of #layers (usually 13) distinct embeddings.
		Each embedding will be processed by the layer-corresponding classifier.
		The specific classifiers and their structure depends on the subclass of the GenderClassifier.

		:param embeddings: A numpy array of dimensions [# samples, # layers (usually = 13), # features (= 768)]
		:return: A numpy array of predictions for each sample and for each layer. The array has dimensions [# samples, # layers]
		The predictions follows the class in the Gender Enumeration.
		"""
		self._check_classifiers_method("predict")
		predictions = np.zeros(shape=(len(embeddings), self.num_layers), dtype=np.uint8)
		for layer in range(self.num_layers):
			clf = self._classifiers[layer]
			predictions[:, layer] = clf.predict(embeddings[:, layer])
		return predictions

	@abstractmethod
	def predict_gender_spectrum(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method computes the "gender score" of the embeddings, similar to a regression model.
		The more the values are centered around zero, the less they're biased. On the other hand, the more the values
		are distant from zero, the more the embeddings are biased towards the male or the female gender.

		Note: a negative value is associated with the male gender. A positive value signals a female gender.

		:param embeddings: A numpy array of dimensions [# samples, # layers (= 13), # features (= 768)]
		:return: A numpy array of gender scores for each sample and for each layer. The array has dimensions [# samples, # layers]
		"""
		pass

	def compute_gender_relevance(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method computes the "gender relevance" of the embeddings.
		Generally speaking, this measure is the scalar product (dot product) between the absolute value of the
		embedding vector and the gender "importance features array".
		In other words, this product scales every component of the embedding according to the importance of that feature
		in the gender definition.

		This function aims to evaluate the efficacy of the classifiers for the given embeddings.

		:param embeddings: A numpy array of dimensions [# samples, # layers (= 13), # features (= 768)]
		:return: A numpy array of intensities for each sample and for each layer. The array has dimensions [# samples, # layers]
		"""
		projections = np.zeros(shape=(len(embeddings), self.num_layers), dtype=np.float)
		for layer in range(self.num_layers):
			features_importance_for_current_layer = self.features_importance[layer]
			projections[:, layer] = np.dot(np.abs(embeddings[:, layer]), features_importance_for_current_layer)
		return projections


class GenderLinearSupportVectorClassifier(_AbstractGenderClassifier):

	def _instance_new_classifier(self):
		return svm.LinearSVC(dual=False)

	@property
	def features_importance(self) -> np.ndarray:
		return np.abs(self.coefficients)

	@property
	def features_bias(self) -> np.ndarray:
		return self.coefficients

	@property
	def coefficients(self) -> np.ndarray:
		"""
		:return: The coefficients array for each layer.
		"""
		return np.asarray([clf.coef_[0] for clf in self._classifiers])

	@property
	def intercepts(self) -> np.ndarray:
		"""
		:return: The intercept value for each layer.
		"""
		return np.asarray([clf.intercept_[0] for clf in self._classifiers])

	def predict_gender_spectrum(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		This method computes the "gender score" of the embeddings, similar to a regression model.
		The more the values are centered around zero, the less they're biased. On the other hand, the more the values
		are distant from zero, the more the embeddings are biased towards the male or the female gender.

		Note: a negative value is associated with the male gender. A positive value signals a female gender.

		In this particular case of Linear SVC, we compute the projection of the embedding over the gender direction.
		The projection of the embedding has several meaning:

		- geometrically, it's the ratio between the projection of the embedding over the gender direction, and the gender
		  direction itself.
		- (also) geometrically, the relative distance of the embedding/point in the space from the hyperspace that
		  divides the two classified classes of gender.
		- algebraically, the scalar product between the embedding and the gender direction, plus the intercept.

		The gender direction corresponds to the coefficients of the trained SVC.
		Each embedding is considered as a group of #layers (=13) distinct embeddings. Each embedding will be
		processed by the layer-corresponding SVC.

		:param embeddings: A numpy array of dimensions [# samples, # layers (= 13), # features (= 768)]
		:return: A numpy array of projections for each sample and for each layer. The array has dimensions [# samples, # layers]
		"""
		projections = np.zeros(shape=(len(embeddings), self.num_layers), dtype=np.float)
		for layer, (coefs, intercept) in enumerate(zip(self.coefficients, self.intercepts)):
			projections[:, layer] = np.dot(embeddings[:, layer], coefs) + intercept
		return projections


class GenderDecisionTreeClassifier(_AbstractGenderClassifier):

	def _instance_new_classifier(self):
		return DecisionTreeClassifier()

	@property
	def features_importance(self) -> np.ndarray:
		return np.asarray([clf.feature_importances_ for clf in self._classifiers])

	@property
	def features_bias(self) -> np.ndarray:
		return self.thresholds[0]

	@property
	def thresholds(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Returns a tuple of the threshold of the features and the
		:return:
		"""
		results = np.zeros(shape=self.features_importance.shape)
		mask = np.zeros(shape=results.shape, dtype=np.uint8)

		for layer in range(self.num_layers):
			clf = self._classifiers[layer]
			features = clf.tree_.feature
			thresholds = clf.tree_.threshold
			for feature, threshold in zip(features, thresholds):

				results[layer, feature] = threshold
				mask[layer, feature] = 1.0
		return results, mask

	def predict_gender_spectrum(self, embeddings: np.ndarray) -> np.ndarray:
		# Input:                [#samples, #layers, #features]
		# Thresholds / Mask:    [#layers, #features]
		# Output:               [#samples, #layers]

		# The gender score is computed with this idea: we take the threshold of the nodes where the decisions happen.
		# Those features are the only one considered for each layer.
		# We compute the distance from the embeddings values to the threshold. With this method, if a feature is "near"
		# the threshold, the result will be low. Otherwise, the result will be high.
		# We sum the distances for each layer and the result is an indication of how "neat" was the classification.
		# At the end, we multiply the result for the predicted class (+1 = female, -1 = male).
		thresholds, mask = self.thresholds
		distances = np.asarray(list(map(
			lambda sample_tensor: np.sum(np.multiply(np.abs(sample_tensor - thresholds), mask), axis=-1),
			embeddings)))
		# Create a +/-1 array of predictions
		classes = self.predict_gender_class(embeddings)
		classes = np.interp(classes, (np.min(classes), np.max(classes)), (-1, +1))
		return np.multiply(distances, classes)


	"""
	test_xs = embeddings[:, layer]
	
	clf = self._classifiers[layer]
	
	feature = clf.tree_.feature
	threshold = clf.tree_.threshold
	
	node_indicator = clf.decision_path(test_xs)
	leaf_id = clf.apply(test_xs)
	
	for sample_id in range(len(embeddings)):
		# obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
		node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]
	
		# print("Rules used to predict sample {id}:\n".format(id=sample_id))
		tot_distance = 0
		for node_id in node_index:
			# continue to the next node if it is a leaf node
			if leaf_id[sample_id] == node_id:
				continue
			# check if value of the split feature for sample 0 is below threshold
			if test_xs[sample_id, feature[node_id]] <= threshold[node_id]:
				threshold_sign = "<="
			else:
				threshold_sign = ">"
	
			print(
				"decision node {node} : (X_test[{sample}, {feature}] = {value}) "
				"{inequality} {threshold})".format(
					node=node_id,
					sample=sample_id,
					feature=feature[node_id],
					value=test_xs[sample_id, feature[node_id]],
					inequality=threshold_sign,
					threshold=threshold[node_id],
				)
			)

	embedding_feature_value = test_xs[sample_id, feature[node_id]]
	node_feature_threshold = threshold[node_id]
	distance = abs(embedding_feature_value - node_feature_threshold)
	tot_distance += distance
	
	results[sample_id, layer] = tot_distance
	"""


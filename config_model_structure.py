import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs
import graph_measures


def generate_random_degree_sequence(n, max_degree=None):
	if not max_degree:
		max_degree = n

	degree_sequence = np.random.randint(1, max_degree, size=n)

	# Need an even number of stubs.
	if sum(degree_sequence)%2 != 0:
		degree_sequence[np.random.randint(n)] += 1

	return degree_sequence


# Generate random configuration model with
# random degree sequence of length n
def generate_random_config_model(n, max_degree=None):
	degree_sequence = generate_random_degree_sequence(n, max_degree)
	return random_graphs.configuration_model(degree_sequence, cleaned=True)


# If performing many computations across graphs do not use this function
# as it is highly inefficient in memory.
def generate_multiple_config_models(amt, n, max_degree=None):
	graphs = []
	for i in xrange(amt):
		graphs.append(generate_random_config_model(n, max_degree))
	return graphs


def analyze_structure_for_fixed_degree_seq(n, max_degree=None):
	# Not sure how to aggregate this into one graph for many different
	# degree sequences

	mean_degrees = []
	mean_neighbor_degrees = []
	# Note, this is the diameter of the largest connected component
	diameters = []
	components_count = []
	global_clustering_coefficients = []
	largest_component_sizes = []

	degree_sequence = generate_random_degree_sequence(n, max_degree)
	for i in xrange(500):
		G = random_graphs.configuration_model(degree_sequence, cleaned=False)
		print (i)

		mean_degrees.append(graph_measures.mean_degree(G))
		mean_neighbor_degrees.append(graph_measures.mean_neighbor_degree(G))
		diameters.append(graph_measures.diameter(G))
		components_count.append(graph_measures.connected_components_count(G))
		global_clustering_coefficients.append(graph_measures.global_clustering_coefficient(G))
		largest_component_sizes.append(graph_measures.largest_component(G))

	# Graph results
	plt.figure(1)

	plt.subplot(234)
	plt.hist(diameters)
	plt.title("Diameter")

	plt.subplot(235)
	plt.hist(components_count)
	plt.title("Components Count")

	plt.subplot(233)
	plt.hist(global_clustering_coefficients)
	plt.title("Clustering Coefficient")

	plt.subplot(231)
	plt.hist(mean_degrees)
	plt.title("Mean Degree")

	plt.subplot(232)
	plt.hist(mean_neighbor_degrees)
	plt.title("Mean Neighbor Degree")

	plt.subplot(236)
	plt.hist(largest_component_sizes)
	plt.title("Largest Component Size")

	plt.show()



def complete_analysis_config_model():
	pass
	# for all sizes up to n, or a few different sizes
	# for many types of max degree

	# can we observe a trend



if __name__ == "__main__":
	analyze_structure_for_fixed_degree_seq(500, 51) # Could set equivlent to erdos renyi expectations


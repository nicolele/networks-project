import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs
import graph_measures



def analyze_structural_identity(rg_generator, trials):
	# Not sure how to aggregate this into one graph for many different
	# degree sequences

	mean_degrees = []
	mean_neighbor_degrees = []
	# Note, this is the diameter of the largest connected component
	diameters = []
	components_count = []
	global_clustering_coefficients = []
	largest_component_sizes = []
	coefficients_of_variations = []

	for i in xrange(trials):
		G = rg_generator()
		#print G.number_of_edges()
		print "Trial, ", i

		mean_degrees.append(graph_measures.mean_degree(G))
		mean_neighbor_degrees.append(graph_measures.mean_neighbor_degree(G))
		diameters.append(graph_measures.diameter(G))
		components_count.append(graph_measures.connected_components_count(G))
		global_clustering_coefficients.append(graph_measures.global_clustering_coefficient(G))
		largest_component_sizes.append(graph_measures.largest_component(G))
		coefficients_of_variations.append(graph_measures.coefficient_of_variation(G))
	
	# Graph results
	plt.figure(1)

	plt.subplot(334)
	plt.hist(diameters, 20)
	plt.title("Diameter")

	plt.subplot(335)
	plt.hist(components_count, 20)
	plt.title("Components Count")

	plt.subplot(333)
	plt.hist(global_clustering_coefficients, 20)
	plt.title("Clustering Coefficient")

	plt.subplot(331)
	plt.hist(mean_degrees, 20)
	plt.title("Mean Degree")

	plt.subplot(332)
	plt.hist(mean_neighbor_degrees, 20)
	plt.title("Mean Neighbor Degree")

	plt.subplot(336)
	plt.hist(largest_component_sizes, 20)
	plt.title("Largest Component Size")

	plt.subplot(337)
	plt.hist(coefficients_of_variations, 20)
	plt.title("Coefficient of Variation")

	plt.show()


# @Wilder, @Nicole
# Use the functions below to pass into function above, see main function
# at bottom for example. You can choose to let the graph be completely
# random, (holding nothing fixed except number of nodes), or you can
# choose many fixed options for the given models. If we mess around with
# these functions, we should be able to come up with 'fingerprint'
# identities for the structure of the different types of graphs.


# Can generate with a fixed p or across many valid options of p
def erdos_renyi_generator(n=500, p=-1):
	if p < 0:
		p = np.random.rand()
	return random_graphs.erdos_renyi(n, p)


def planted_partition_generator(n=500, groups=3, pin=-1, pout=-1, predefined_communities=[]):
	# Option for random pins and pouts with pins guaranteed to be > 0.5
	# and pouts to be < 0.5
	if pin < 0:
		pin = np.random.rand()
		if pin < 0.5:
			pin = 1 - pin
	if put < 0:
		pout = np.random.rand()
		if pout > 0.5:
			pout = 1 - pout

	# random sized communities as long as number of groups is fixed
	if len(predefined_communities) == 0:
		communities = []
		for i in xrange(groups-1):
			# Add an element of randomness so some graphs have even splits
			# and some do not
			c_size = np.random.randint(int(n * np.random.rand()))
			communities.append(c_size)
			n -= c_size
		communities.append(n)
	else:
		communities = predefined_communities

	#print communities
	return random_graphs.random_partition_model(communities, pin, pout)


def barabasi_albert_generator(n=500, c=3):
	if c < 0:
		c = np.random.randint(20)
	#print c
	return random_graphs.barabasi_albert_model(n, c)


def geometric_generator(n=500, r=-1):
	if r < 0:
		r = np.random.rand()
	#print r
	return random_graphs.geometric_model(n, r)


def watts_strogatz_generator(n=500, k=3, p=-1):
	if k < 0:
		k = np.random.randint(20)
	if p < 0:
		p = np.random.rand()
	#print k, p
	return random_graphs.watts_strogatz_model(n, k, p)


def configuration_model_generator(n=500, max_degree=-1, fixed_sequence = []):
	if len(fixed_sequence) != 0:
		return random_graphs.configuration_model(fixed_sequence, cleaned=True)
	
	if max_degree < 0:
		max_degree = np.random.randint(1, n)
	#print max_degree

	degree_sequence = np.random.randint(1, max_degree, size=n)
	return random_graphs.configuration_model(degree_sequence, cleaned=True)



if __name__ == "__main__":
	analyze_structural_identity(configuration_model_generator, 25)



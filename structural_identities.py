import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs
import graph_measures
import matplotlib.mlab as mlab


MAX_ATTEMPTS = 100
CONSTRAINTS_LOOKUP = {
 'diameter': graph_measures.diameter,
 'mean_degree': graph_measures.mean_degree,
 'mean_neighbor_degree': graph_measures.mean_neighbor_degree,
 'global_clustering_coefficient': graph_measures.global_clustering_coefficient,
 'connected_components_count': graph_measures.connected_components_count,
 'largest_component': graph_measures.largest_component,
 'coefficient_of_variation': graph_measures.coefficient_of_variation
}


def analyze_structural_identity(rg_generator, trials, fig, constraints=None):
	# Not sure how to aggregate this into one graph for many different
	# degree sequences
	print constraints

	mean_degrees = []
	mean_neighbor_degrees = []
	# Note, this is the diameter of the largest connected component
	diameters = []
	components_count = []
	global_clustering_coefficients = []
	largest_component_sizes = []
	coefficients_of_variations = []
	low_centrals = []
	high_centrals = []

	for i in xrange(trials):
		G = constrained_generation(rg_generator, constraints)
		#print G.number_of_edges()
		print "Trial, ", i

		mean_degrees.append(graph_measures.mean_degree(G))
		mean_neighbor_degrees.append(graph_measures.mean_neighbor_degree(G))
		diameters.append(graph_measures.diameter(G))
		components_count.append(graph_measures.connected_components_count(G))
		global_clustering_coefficients.append(graph_measures.global_clustering_coefficient(G))
		largest_component_sizes.append(graph_measures.largest_component(G))
		coefficients_of_variations.append(graph_measures.coefficient_of_variation(G))
		centrals = graph_measures.hi_lo_centrality(graph_measures.betweenness_centrality, G)
		low_centrals.append(centrals[0])
		high_centrals.append(centrals[1])


	# Graph results
	plt.figure(fig)

	kwargs = dict(histtype='stepfilled', alpha=0.5, normed=True, bins=20)

	dic = {331:"Mean Degree", 332:"Mean Neighbor Degree",\
		   333:"Clustering Coefficient", 334:"Diameter",\
		   335:"Components Count", 336:"Largest Component Size",\
		   337:"Coefficient of Variation", 338:"Smallest Betweenness Centrality",\
		   339:"Largest Betweenness Centrality"}


	plt.subplot(334)
	plt.hist(diameters, **kwargs)
	plt.title("Diameter")

	plt.subplot(335)
	plt.hist(components_count, **kwargs)
	plt.title("Components Count")

	plt.subplot(333)
	plt.hist(global_clustering_coefficients, **kwargs)
	plt.title("Clustering Coefficient")

	plt.subplot(331)
	plt.hist(mean_degrees, **kwargs)
	plt.title("Mean Degree")

	plt.subplot(332)
	plt.hist(mean_neighbor_degrees, **kwargs)
	plt.title("Mean Neighbor Degree")

	plt.subplot(336)
	plt.hist(largest_component_sizes, **kwargs)
	plt.title("Largest Component Size")

	plt.subplot(337)
	plt.hist(coefficients_of_variations, **kwargs)
	plt.title("Coefficient of Variation")

	plt.subplot(338)
	plt.hist(low_centrals, **kwargs)
	plt.title("Smallest Betweenness Centrality")

	plt.subplot(339)
	plt.hist(high_centrals, **kwargs)
	plt.title("Largest Betweenness Centrality")
	plt.tight_layout()

	plt.show()

	return [mean_degrees, mean_neighbor_degrees, global_clustering_coefficients,\
			diameters, components_count, largest_component_sizes,\
			coefficients_of_variations, low_centrals, high_centrals], dic


def graph_constrained_distributions(fig, points, points_constrained, dic):
	plt.figure(fig)

	for i in xrange(len(points)):
		bins = np.histogram(np.hstack((points[i],points_constrained[i])), bins=20)[1] #get the bin edges
		kwargs = dict(histtype='stepfilled', alpha=0.5, bins=bins)
		
		plt.subplot(331+i)
		plt.hist(points[i], **kwargs)
		plt.hist(points_constrained[i], **kwargs)
		plt.title(dic[331+i])

	plt.tight_layout()
	plt.show()

N = 250

# Can generate with a fixed p or across many valid options of p
def erdos_renyi_generator(n=N, p=-1):
	if p < 0:
		p = np.random.rand()
	return random_graphs.erdos_renyi(n, p)


def planted_partition_generator(n=N, groups=3, pin=-1, pout=-1, predefined_communities=[]):
	# Option for random pins and pouts with pins guaranteed to be > 0.5
	# and pouts to be < 0.5
	if pin < 0:
		pin = np.random.rand()
		if pin < 0.5:
			pin = 1 - pin
	if pout < 0:
		pout = np.random.rand()
		if pout > 0.5:
			pout = 1 - pout

	# random sized communities as long as number of groups is fixed
	if len(predefined_communities) == 0:
		communities = []
		for i in xrange(groups-1):
			# Add an element of randomness so some graphs have even splits
			# and some do not
			c_size = np.random.randint(1 + int(n * np.random.rand()))
			communities.append(c_size)
			n -= c_size
		communities.append(n)
	else:
		communities = predefined_communities

	#print communities
	return random_graphs.random_partition_model(communities, pin, pout)


def barabasi_albert_generator(n=N, c=-1):
	if c < 0:
		c = np.random.randint(1, 50)
	#print c
	return random_graphs.barabasi_albert_model(n, c)


def geometric_generator(n=N, r=-1):
	if r < 0:
		r = np.random.rand()
		if r < 0.1:
			r = 1 - r
	#print r
	return random_graphs.geometric_model(n, r)


def watts_strogatz_generator(n=N, k=-1, p=-1):
	if k < 0:
		k = np.random.randint(5, 25)
	if p < 0:
		p = np.random.rand()
	#print k, p
	return random_graphs.watts_strogatz_model(n, k, p)


def configuration_model_generator(n=N, max_degree=-1, fixed_sequence = []):
	if len(fixed_sequence) != 0:
		return random_graphs.configuration_model(fixed_sequence, cleaned=True)
	
	if max_degree < 0:
		max_degree = np.random.randint(2, 100)
	#print max_degree

	degree_sequence = np.random.randint(1, max_degree, size=n)
	return random_graphs.configuration_model(degree_sequence, cleaned=True)


# Constraints dictionary should be of the form
# <name of constraint according to above dictionary, tuple 
# represeting valid range>
def satisfies_constraints(G, constraints):
	for constraint, valid_range in constraints.items():

		value = CONSTRAINTS_LOOKUP[constraint](G)
		if not value <= valid_range[1] or not value >= valid_range[0]:
			print value
			return False

	return True


# Constraints is a dictionary
def constrained_generation(generator_function, constraints):
	if not constraints:
		return generator_function()

	attempts = 0
	while attempts < MAX_ATTEMPTS:
		G = generator_function()
		print "OK"
		if satisfies_constraints(G, constraints):
			return G
		
		attempts += 1

	print "Desired function cannot meet constraints... exiting."


if __name__ == "__main__":
	points, dic = analyze_structural_identity(configuration_model_generator, 250, 1) # Fig 1
	#analyze_structural_identity(watts_strogatz_generator, 1000, 2)
	#analyze_structural_identity(geometric_generator, 1000, 3)
	#analyze_structural_identity(erdos_renyi_generator, 1000, 4)
	#analyze_structural_identity(barabasi_albert_generator, 1000, 5)
	#analyze_structural_identity(planted_partition_generator, 1000, 6) # Figt 6

	constraints = {'diameter': (4, 5)}
	constrained_points, dic = analyze_structural_identity(configuration_model_generator, 250, 1, constraints)

	graph_constrained_distributions(1, points, constrained_points, dic)



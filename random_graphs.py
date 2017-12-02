from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# Inputs:
# 	n - number of nodes
#	p - probability that an edge i,j will exist
# Returns:
# 	A networkx random graph with the above properties
#
# For more information and erdos renyi graph properties....
# 	http://tuvalu.santafe.edu/~aaronc/courses/5352/csci5352_2017_L3.pdf
def erdos_renyi(n, p):
	return nx.erdos_renyi_graph(n, p)


# Inputs:
# 	degree_sequence - a list of desired degrees for given nodes
#	cleaned - removes self-edges and duplicate edges
# Returns:
# 	A networkx random graph with the above properties
def configuration_model(degree_sequence, cleaned=False):
	if sum(degree_sequence)%2 != 0:
		print ("Cannot create model with odd number of stubs.")
		exit()

	rg = nx.Graph(nx.configuration_model(degree_sequence))
	if cleaned:
		rg.remove_edges_from(rg.selfloop_edges())

	return rg

# Inputs:
# 	community_sizes - an array specifying number of communities
#		and size of each. [10, 15] implies 2 groups, sizes 10 and 15
#	pin - probability of two nodes within a group to be connected
#	pout - probability of two nodes across groups to be connected
# Returns:
# 	A networkx random graph with the above properties
def random_partition_model(community_sizes, pin, pout):
	return nx.random_partition_graph(community_sizes, pin, pout)


# Inputs:
# 	n - number of nodes placed randomly in the unit cube
#	r - a distance for which nodes less than r apart should connect
#		in a euclidean space
# Returns:
# 	A networkx random graph with the above properties
def geometric_model(n, r):
	return nx.random_geometric_graph(n, r)


# Inputs:
# 	n - number of nodes
#	c - number of edges each newcomer node should have
#
#	The model displays the preferential attachment idea.
# Returns:
# 	A networkx random graph with the above properties
def barabasi_albert_model(n, c):
	return nx.barabasi_albert_graph(n, c)


# Inputs:
# 	n - number of nodes
#	k - each node is connected to its k nearest neighbors
#	p - probability of adding a new edge for each edge
#
#	The model displays the small-world effect
# Returns:
# 	A networkx random graph with the above properties
#
# More information here
#	https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html#networkx.generators.random_graphs.watts_strogatz_graph
def watts_strogatz_model(n, k, p):
	return nx.watts_strogatz_graph(n, k, p)




if __name__ == "__main__":
	pass
	# Your Test Here



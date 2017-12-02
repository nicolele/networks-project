from __future__ import division

import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs


def sort_centrality(centralities):
	return sorted(centralities.items(), key=operator.itemgetter(1), reverse=True)


def degree_centrality(G):
	return nx.degree_centrality(G)


def harmonic_centrality(G):
	return nx.harmonic_centrality(G)


def eigenvector_centrality(G):
	return nx.eigenvector_centrality(G)


def betweenness_centrality(G):
	return nx.betweenness_centrality(G)


def diameter(G):
	return nx.diameter(G)


def mean_degree(G):
	degrees = G.degree()
	return sum(degrees.values())/len(degrees)


def global_clustering_coefficient(G):
	return nx.average_clustering(G)


# Get the local clustering coefficient for a list of node id's passed in
def local_clustering_coefficient(G, nodes=[]):
	if len(nodes) > 0:
		return nx.clustering(G, nodes)
	return nx.clustering(G)


def connected_components_count(G):
	return nx.number_connected_components(G)


# Skip assortativity for now as we dont have labeled nodes
def assortativity(G):
	pass


if __name__ == "__main__":
	pass
	


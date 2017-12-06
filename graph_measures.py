from __future__ import division

import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs
from math import sqrt

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

# Diameter of largest component
def diameter(G):
	if connected_components_count(G) > 1:
		nodes = max(nx.connected_components(G), key=len)

		new_G = nx.Graph()

		for edge in G.edges():
			if edge[0] in nodes and edge[1] in nodes:
				new_G.add_edge(edge[0], edge[1])

		G = new_G
	
	return nx.diameter(G)

def mean_degree(G):
	degrees = G.degree()
	return sum(degrees.values())/len(degrees)


def mean_neighbor_degree_per_node(G):
	return nx.average_neighbor_degree(G)


def mean_neighbor_degree(G):
	neighbor_degrees = mean_neighbor_degree_per_node(G)
	return sum(neighbor_degrees.values())/len(neighbor_degrees)


def global_clustering_coefficient(G):
	return nx.average_clustering(G)


# Get the local clustering coefficient for a list of node id's passed in
def local_clustering_coefficient(G, nodes=[]):
	if len(nodes) > 0:
		return nx.clustering(G, nodes)
	return nx.clustering(G)


def connected_components_count(G):
	return nx.number_connected_components(G)


def largest_component(G):
	return len(max(nx.connected_components(G), key=len))

def degree_assortativity(G):
	return nx.degree_assortativity_coefficient(G)

def coefficient_of_variation(G):
	mean = sum(G.degrees())/len(G.degrees())
	sd = 0
	for degree_k in G.degrees():
		sd += (degree_k-mean)**2
	sd = sd/(len(G.degrees())-1)
	sd = sqrt(sd)
	return sd/mean

#if __name__ == "__main__":
	#G = random_graphs.erdos_renyi(500, 0.1)
	#print(mean_neighbor_degree(G))
	#print(mean_degree(G))
	



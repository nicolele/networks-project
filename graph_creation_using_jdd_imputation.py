import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs
import graph_measures
import edge_imputation
import structural_identities
from collections import defaultdict
from numpy.linalg import norm
from copy import deepcopy


def jdd(G):
	degrees = G.degree()
	jdd = defaultdict(int)

	for edge in G.edges():
		d1 = degrees[edge[0]]
		d2 = degrees[edge[1]]
		jdd[(d1, d2)] += 1

	return jdd


def graph_difference(G1, G2, jdd_1=None, jdd_2=None):
	if jdd_1 is None or jdd_2 is None:
		jdd_1 = jdd(G1)
		jdd_2 = jdd(G2)
	else:
		jdd_1 = deepcopy(jdd_1)
		jdd_2 = deepcopy(jdd_2)

	m = max(max(G1.degree().values()), max(G2.degree().values()))

	M = [[0 for i in xrange(m)] for j in xrange(m)]
	for i in xrange(m):
		for j in xrange(m):
			M[i][j] = jdd_1[(i, j)] - jdd_2[(i, j)]

	return norm(M, 'fro')


# Does not work
def impute_edge_algorithm(G, target_G):
	# Given a graph, add edges to minimize to the desired JDD
	target_jdd = jdd(target_G)

	nodes = G.nodes()
	degrees = G.degree()
	
	current_jdd = jdd(G)
	current_diff = graph_difference(G, target_G, current_jdd, target_jdd)

	for i in nodes:
		for j in nodes:
			if not G.has_edge(i, j):
				d1 = degrees[i]	+ 1
				d2 = degrees[j] + 1

				current_jdd[(d1, d2)] += 1
				
				if target_jdd.has_key((d1, d2)):
					temp_diff = graph_difference(G, target_G, current_jdd, target_jdd)
				
					if temp_diff <= current_diff:
						current_diff = temp_diff
						G.add_edge(i, j)
						print temp_diff
					else:
						current_jdd[(d1, d2)] -= 1
				else:
					G.add_edge(i, j)

		print current_diff

	print G.number_of_edges(), target_G.number_of_edges()


def test_edge_imputation():
	G = random_graphs.barabasi_albert_model(500, 10)
	degree_sequence = [1] * 500

	new_G = random_graphs.configuration_model(degree_sequence)

	impute_edge_algorithm(new_G, G)


def produce_jdds_for_rg(rg_generator):
	# Create a bunch of trials and average to get a jdd
	#analyze_structural_identity(configuration_model_generator, 100, 1) # Fig 1
	#analyze_structural_identity(watts_strogatz_generator, 1000, 2)
	#analyze_structural_identity(geometric_generator, 1000, 3)
	#analyze_structural_identity(erdos_renyi_generator, 1000, 4)
	#analyze_structural_identity(barabasi_albert_generator, 1000, 5)
	#analyze_structural_identity(planted_partition_generator, 1000, 6)
	pass


def predict_structure(G):
	# Given a graph, predict what type of structure it has
	# Do the impute edge_algorithm if need be
	# call produce_jdds_for_rg for each generator
	pass


def compare_statistics(G1, G2):
	pass 



def run_graph_matching():
	# Create a bunch of trials and average to get a jdd
	#rgs = [#[structural_identities.configuration_model_generator,
	rgs = [structural_identities.watts_strogatz_generator(500, 22),
		   structural_identities.geometric_generator(500, 0.125),
		   structural_identities.erdos_renyi_generator(500, 0.05),
		   structural_identities.barabasi_albert_generator(500, 12),
		   structural_identities.planted_partition_generator(500, 3, 0.1, 0.01)]

	rgs2 = [structural_identities.watts_strogatz_generator(500, 22),
		   structural_identities.geometric_generator(500, 0.125),
		   structural_identities.erdos_renyi_generator(500, 0.05),
		   structural_identities.barabasi_albert_generator(500, 12),
		   structural_identities.planted_partition_generator(500, 3, 0.1, 0.01)]


	samples = 150
	matrix_diff = [[0 for i in xrange(len(rgs))] for j in xrange(len(rgs))]
	for sample in xrange(samples):
		print sample
		for i, rg1 in enumerate(rgs):
			for j, rg2 in enumerate(rgs2):
				matrix_diff[i][j] += (graph_difference(rg1, rg2)**2)

	for i in xrange(len(matrix_diff)):
		for j in xrange(len(matrix_diff[i])):
			matrix_diff[i][j] /= samples

	return matrix_diff


def plot_graph_matching():
	m_diff = run_graph_matching()

	objects = ('WS', 'Geo', 'ER', 'BA', 'PPM')
	y_pos = np.arange(len(objects))

	plt.figure(1)
	plt.tight_layout()
	# plt.subplot(231)
	# plt.bar(y_pos, m_diff[0], align='center', alpha=0.5)
	# plt.xticks(y_pos, objects)
	# plt.ylabel('Graph Difference (JDD Norms)')
	# plt.title('Configuration Model')

	plt.subplot(231)
	plt.bar(y_pos, m_diff[0], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Graph Difference (JDD Norms)')
	plt.title('Watts Strogatz')

	plt.subplot(232)
	plt.bar(y_pos, m_diff[1], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	#plt.ylabel('Graph Difference (JDD Norms)')
	plt.title('Geometric')

	plt.subplot(233)
	plt.bar(y_pos, m_diff[2], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	#plt.ylabel('Graph Difference (JDD Norms)')
	plt.title('Erdos Renyi')

	plt.subplot(234)
	plt.bar(y_pos, m_diff[3], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Graph Difference (JDD Norms)')
	plt.title('Barabasi Albert')

	plt.subplot(235)
	plt.bar(y_pos, m_diff[4], align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	#plt.ylabel('Graph Difference (JDD Norms)')
	plt.title('Planted Partition')

	plt.show()



if __name__ == "__main__":
	# G = random_graphs.barabasi_albert_model(500, 10)
	# G2 = random_graphs.barabasi_albert_model(500, 10)
	# G3 = random_graphs.erdos_renyi(500, 0.03)
	# G5 = random_graphs.erdos_renyi(500, 0.03)
	# G4 = random_graphs.geometric_model(500, 0.12)
	# print G4.number_of_edges(), G3.number_of_edges()
	plot_graph_matching()
	#test_edge_imputation()



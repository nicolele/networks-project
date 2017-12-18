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
import seaborn as sns



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



def run_graph_matching():
	# Create a bunch of trials and average to get a jdd
	#rgs = [#[structural_identities.configuration_model_generator,

	samples = 150
	matrix_diff = [[0 for i in xrange(5)] for j in xrange(5)]

	for sample in xrange(samples):
		print sample
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



# Pass in G
def predict_structure(G, trials=20, constraints=False):
	n = G.number_of_nodes()
	e = G.number_of_edges()
	rgs = [structural_identities.watts_strogatz_generator,
		   structural_identities.geometric_generator,
		   structural_identities.erdos_renyi_generator,
		   structural_identities.barabasi_albert_generator,
		   structural_identities.planted_partition_generator]
	index = ['Watts Strogatz', 'Geometric', 'Erdos Renyi', 'Barabasi Albert', 'Planted Partition Model']

	if not constraints:
		constraints = {'edge_count': (.75*e, 1.25*e)}
	else:
		diam = graph_measures.diameter(G)
		#md = graph_measures.mean_degree(G)
		constraints = {'edge_count': (.75*e, 1.25*e),
					   'diameter': (.1 * diam, 2*diam)}
					   #'mean_degree':(.5*md, 1.5*md)}


	difs = [0 for x in xrange(len(rgs))]
	for _ in xrange(trials):
		print _
		temp = []
		for i, rg in enumerate(rgs):
			G_2 = structural_identities.constrained_generation(rg, constraints)
			dif = graph_difference(G, G_2)
			temp.append(dif)
		array = np.array(temp)
		order = array.argsort()
		ranks = order.argsort()

		for i in xrange(len(ranks)):
			difs[i] += ranks[i]


	total = sum(difs)*1.0
	for i in xrange(len(difs)):
		difs[i] = difs[i]/total

	# for i in xrange(len(difs)):
	# 	print difs[i], index[i]

	return difs, index


def run_predict_structure(generator=None, title=None):
	constraints = {'edge_count': (1000, 1100)}

	accuracy_at_k = [0] * 5

	if generator != None and title != None:
		samples = 100
		for sample in xrange(samples):
			G = structural_identities.constrained_generation(generator, constraints)
			cluster, types = predict_structure(G, trials=20)

			print sample, types[cluster.index(min(cluster))]
		
			array = np.array(cluster)
			order = array.argsort()
			ranks = order.argsort().tolist()

			k = -1
			for i in xrange(len(cluster)): # 5 types of rg
				if title==types[ranks.index(i)]:
					k = i
					break

			j = len(cluster)-1
			while j >= k:
				accuracy_at_k[j] += 1
				j -= 1


		plt.figure(1)

		for i in xrange(len(accuracy_at_k)):
			accuracy_at_k[i] /= (samples*1.0)

		plt.plot([i for i in xrange(1, 6)], accuracy_at_k, marker='o')
		plt.xlabel('k (top k labels)')
		plt.ylim((0, 1.1))
		plt.ylabel('Accuracy @ k')
		plt.title('Prediction Accuracy for ' + title + ' Random Graphs')
			
		plt.show()

	# Uniformly sample across rg
	elif generator == None:
		confusion_matrix = [[0 for i in xrange(5)] for j in xrange(5)]
		samples = 100
		index = ['Watts Strogatz', 'Geometric', 'Erdos Renyi', 'Barabasi Albert', 'Planted Partition Model']
		constraints_enforced=True
		rgs = [structural_identities.watts_strogatz_generator,
			   structural_identities.geometric_generator,
			   structural_identities.erdos_renyi_generator,
			   structural_identities.barabasi_albert_generator,
			   structural_identities.planted_partition_generator]


		for j, rg in enumerate(rgs):
			title = index[j]
			actual = j
			for i in xrange(samples):
				G = structural_identities.constrained_generation(rg, constraints)
				
				cluster, types = predict_structure(G, 5, constraints_enforced)

				predicted = cluster.index(min(cluster))
				print title, types[predicted]

				confusion_matrix[actual][predicted] += 1

				array = np.array(cluster)
				order = array.argsort()
				ranks = order.argsort().tolist()

				k = -1
				for i in xrange(len(cluster)): # 5 types of rg
					if title==types[ranks.index(i)]:
						k = i
						break

				j = len(cluster)-1
				while j >= k:
					accuracy_at_k[j] += 1
					j -= 1

		small_index = ['WS', 'Geo', 'ER', 'BA', 'PPM']


		for i in xrange(len(accuracy_at_k)):
			accuracy_at_k[i] /= (samples*1.0*len(rgs))

		print accuracy_at_k

		if constraints_enforced:
			plt.plot([i for i in xrange(1, 6)], accuracy_at_k, marker='o', color='red')
		else:
			plt.plot([i for i in xrange(1, 6)], accuracy_at_k, marker='o')
		plt.xlabel('k (top k labels)')
		plt.ylim((0, 1.1))
		plt.ylabel('Accuracy @ k')
		plt.title('Prediction Accuracy for Uniformly Sampled Random Graphs')
			
		plt.show()

		sns.set()
		ax = plt.axes()
		sns.heatmap(confusion_matrix, ax = ax, cmap="YlGnBu", yticklabels=index,xticklabels=small_index)
		ax.set_title('Confusion Matrix for Uniformly Sampled Random Graphs')
		plt.tight_layout()
		plt.show()



# Does not work
def impute_edge_algorithm(G, target_G):
	# Given a graph, add edges to minimize to the desired JDD
	target_jdd = jdd(target_G)

	nodes = G.nodes()
	degrees = G.degree()
	
	current_jdd = jdd(G)
	current_diff = graph_difference(G, target_G, current_jdd, target_jdd)

	#exit()
	count = 0
	while (G.number_of_edges() < target_G.number_of_edges()):
		count += 1
		best_edge = (0, 0)
		best_diff = float('inf')
		for _ in xrange(3):
			i = np.random.randint(len(nodes))
			j = np.random.randint(len(nodes))

			if not G.has_edge(i, j):
				d1 = degrees[i]	+ 1
				d2 = degrees[j] + 1

				current_jdd[(d1, d2)] += 1
					
				temp_diff = graph_difference(G, target_G, current_jdd, target_jdd)
				
				if temp_diff < best_diff:
					best_edge = (i, j)
					current_jdd[(d1, d2)] -= 1

		G.add_edge(*best_edge)

	return G


def test_edge_imputation():
	constraints = {'edge_count': (1000, 1100)}
	accuracy_at_k = [0] * 5

	confusion_matrix = [[0 for i in xrange(5)] for j in xrange(5)]
	samples = 100
	index = ['Watts Strogatz', 'Geometric', 'Erdos Renyi', 'Barabasi Albert', 'Planted Partition Model']
	constraints_enforced=False
	rgs = [structural_identities.watts_strogatz_generator,
		   structural_identities.geometric_generator,
		   structural_identities.erdos_renyi_generator,
		   structural_identities.barabasi_albert_generator,
		   structural_identities.planted_partition_generator]


	for uni, rg in enumerate(rgs):
		title = index[uni]
		actual = uni
		created_graphs = []
		for i in xrange(samples):
			G = structural_identities.constrained_generation(rg, constraints)
			
			degree_sequence = [1] * G.number_of_nodes()

			new_G = random_graphs.configuration_model(degree_sequence)
			new_G = impute_edge_algorithm(new_G, G)
			created_graphs.append(new_G)

			cluster, types = predict_structure(new_G, 2, constraints_enforced)

			predicted = cluster.index(min(cluster))
			print title, types[predicted]

			confusion_matrix[actual][predicted] += 1

			array = np.array(cluster)
			order = array.argsort()
			ranks = order.argsort().tolist()

			k = -1
			for i in xrange(len(cluster)): # 5 types of rg
				if title==types[ranks.index(i)]:
					k = i
					break

			j = len(cluster)-1
			while j >= k:
				accuracy_at_k[j] += 1
				j -= 1

		# HERE we plot distros
		observed_metrics, dic = structural_identities.analyze_structural_identity_graphs(created_graphs, uni)
		predict_metrics, dic = structural_identities.analyze_structural_identity(rg, samples, uni)# constraints=None):
		structural_identities.graph_created_distributions(uni, observed_metrics, predict_metrics, dic)


	small_index = ['WS', 'Geo', 'ER', 'BA', 'PPM']


	plt.figure(10)

	for i in xrange(len(accuracy_at_k)):
		accuracy_at_k[i] /= (samples*1.0*len(rgs))

	if constraints_enforced:
		plt.plot([i for i in xrange(1, 6)], accuracy_at_k, marker='o', color='red')
	else:
		plt.plot([i for i in xrange(1, 6)], accuracy_at_k, marker='o')
	
	plt.xlabel('k (top k labels)')
	plt.ylim((0, 1.1))
	plt.ylabel('Accuracy @ k')
	plt.title('Prediction Accuracy for Uniformly Sampled Random Graphs')
		
	plt.show()

	sns.set()
	ax = plt.axes()
	sns.heatmap(confusion_matrix, ax = ax, cmap="YlGnBu", yticklabels=index,xticklabels=small_index)
	ax.set_title('Confusion Matrix for Uniformly Sampled Random Graphs')
	plt.tight_layout()
	plt.show()



def perform_edge_imputation():
	pass


if __name__ == "__main__":
	test_edge_imputation()
	# G = random_graphs.barabasi_albert_model(500, 10)
	# G2 = random_graphs.barabasi_albert_model(500, 10)
	# G3 = random_graphs.erdos_renyi(500, 0.03)
	# G5 = random_graphs.erdos_renyi(500, 0.03)
	# G4 = random_graphs.geometric_model(500, 0.12)
	# print G4.number_of_edges(), G3.number_of_edges()
	# plot_graph_matching()
	# test_edge_imputation()
	# soft_clustering()
	#run_predict_structure()
	#run_predict_structure(structural_identities.erdos_renyi_generator, 'Erdos Renyi')
	# run_predict_structure(structural_identities.geometric_generator, 'Geometric')
	# run_predict_structure(structural_identities.barabasi_albert_generator, 'Barabasi Albert')
	# run_predict_structure(structural_identities.planted_partition_generator, 'Planted Partition Model')
	# run_predict_structure(structural_identities.watss_strogatz_generator, 'Watts Strogatz')






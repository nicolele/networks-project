import networkx as nx
from graph_measures import degree_assortativity, degree_centrality, harmonic_centrality, eigenvector_centrality
from graph_measures import betweenness_centrality, diameter, mean_degree, mean_neighbor_degree
from graph_measures import coefficient_of_variation, global_clustering_coefficient
from random_graphs import random_partition_model,erdos_renyi, geometric_model, barabasi_albert_model
from joint_degree_investigation import joint_degree_distribution
from numpy.linalg import norm
from numpy.random import choice,uniform
from math import  floor, sqrt
from random import sample
from copy import deepcopy
import random
from numpy import linspace
from matplotlib import pyplot as plt
from numpy.random import randint

def remove_edges(g, p):

    edge_set = set()
    edges = list(g.edges())
    num_edges = g.number_of_edges()

    to_remove = floor(num_edges * p)

    removed = 0
    while removed < int(to_remove):
        random_edge_index = randint(num_edges)
        edge = edges[random_edge_index]
        edge_set.add((edge[0], edge[1]))
        removed += 1

    g.remove_edges_from(list(edge_set))

    return g

def pull_graph_statistics(G):
    statistics = {}
    statistics[degree_centrality] = degree_centrality(G)
    statistics[harmonic_centrality] = harmonic_centrality(G)
    statistics[eigenvector_centrality] = eigenvector_centrality(G)
    statistics[betweenness_centrality] = betweenness_centrality(G)
    statistics[diameter] = diameter(G)
    statistics[mean_degree] = mean_degree(G)
    statistics[mean_neighbor_degree] = mean_neighbor_degree(G)
    statistics[global_clustering_coefficient] = global_clustering_coefficient(G)
    statistics[degree_assortativity] = degree_assortativity(G)
    statistics[coefficient_of_variation] = coefficient_of_variation(G)

    return statistics

def Parse_distribution(statistics_distribution_dict, restrictions_dict):
    current_statistics = {}
    for key in statistics_distribution_dict.keys():
        if key not in restrictions_dict:
            current_statistics[key] = []

    for key in restrictions_dict.keys():
        current_dist = statistics_distribution_dict[key]
        parsed_dist = [stat for stat in current_dist if stat in restrictions_dict[key]]
        current_statistics[key] = parsed_dist

    return current_statistics

def norm_difference_graphs(G_1,G_2, norm_name):
    g_1_max = max(list(G_1.degree().values()))
    g_2_max = max(list(G_1.degree().values()))
    total_degrees = max(g_1_max,g_2_max)
    JDD_1 = joint_degree_distribution(G_1, total_degrees)
    JDD_2 = joint_degree_distribution(G_2, total_degrees)
    M = [[0 for i in range(total_degrees)] for j in range(total_degrees)]
    for index_i in range(total_degrees):
        for index_j in range(total_degrees):
            M[index_i][index_j] = JDD_1[str(index_i) + ',' + str(index_j)] - JDD_2[str(index_i) + ',' + str(index_j)]
    return norm(M,norm_name)

def norm_difference(JDD_1,JDD_2, norm_name):
    total_degrees_1 = int(sqrt(len(JDD_1.keys())))
    total_degrees_2 = int(sqrt(len(JDD_2.keys())))
    total_degrees = max(total_degrees_1, total_degrees_2)
    M = [[0 for i in range(total_degrees)] for j in range(total_degrees)]
    for index_i in range(total_degrees):
        for index_j in range(total_degrees):
            index = str(index_i) + ',' + str(index_j)
            if index in JDD_1.keys() and index in JDD_2.keys():
                M[index_i][index_j] = abs(JDD_1[str(index_i) + ',' + str(index_j)] - JDD_2[str(index_i) + ',' + str(index_j)])
            elif index in JDD_1.keys():
                M[index_i][index_j] = abs(JDD_1[str(index_i) + ',' + str(index_j)])
            elif index in JDD_2.keys():
                M[index_i][index_j] = abs(JDD_2[str(index_i) + ',' + str(index_j)])
            else:
                M[index_i][index_j] = 0

    return norm(M,norm_name)

def simulate_random_graph(graph_name, number_of_nodes):

    if graph_name == "random_partition_model":
        p_in = uniform()
        p_out = uniform()
        community_sizes = uniform_partition_gen(2, number_of_nodes, choice(floor(number_of_nodes / 2)))
        return random_partition_model(community_sizes, p_in, p_out)

    elif graph_name == "erdos_renyi":
        p = uniform()
        return erdos_renyi(number_of_nodes,p)

    elif graph_name == "geometric_model":
        r = floor(sqrt(choice(number_of_nodes))+1)
        return geometric_model(number_of_nodes,r)

    elif graph_name == "barabasi_albert_model":
        c = choice(floor(number_of_nodes/2))+1
        return barabasi_albert_model(number_of_nodes,c)

    else:
        print('please provide a valid graphical model')
        return None

def generate_proposal_JDDs(number_of_nodes, distributions, averaging_per_proposal):
    average_JDD = {}
    for graph in distributions:

        average_JDD[graph] = None
        for sim in range(averaging_per_proposal):
            simulate = 1
            while simulate:
                current_G = simulate_random_graph(graph, number_of_nodes)
                if len(list(current_G.degree().values()))>0:
                    simulate = 0

            total_degrees =  max(list(current_G.degree().values()))
            current_JDD = joint_degree_distribution(current_G,total_degrees)

            if average_JDD[graph] is None:
                average_JDD[graph] = current_JDD
            else:
                keys = list(current_JDD.keys()) + list(average_JDD[graph].keys())
                keys = set(keys)
                for key in keys:
                    if (key in current_JDD.keys()) and (key in average_JDD[graph].keys()):
                        average_JDD[graph][key] = current_JDD[key] + average_JDD[graph][key]
                    elif key in current_JDD.keys():
                        average_JDD[graph][key] = current_JDD[key]
                    elif key in average_JDD[graph].keys():
                        average_JDD[graph][key] = average_JDD[graph][key]
                    else:
                        average_JDD[graph][key] = 0

        for key in average_JDD[graph].keys():
            average_JDD[graph][key] /= averaging_per_proposal

    return average_JDD

def uniform_partition_gen(min_val, sum_total, total_parts):
    M = sum_total - total_parts * (min_val - 1)
    if M < total_parts:
        return
    else:
        check = [ii for ii in range(0, M - 1)]
        perms = sample(check, len(check))
        perms = [x + 1 for x in perms]
        picks = deepcopy(perms[0:total_parts - 1])
        picks.sort()
        picks.append(M)
        picks = [0] + picks
    return [picks[i] - picks[i - 1] + min_val - 1 for i in range(1, total_parts + 1)]

def edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, p):

    number_of_nodes = len(Observered_G.nodes())
    # generate average JDD for proposal distributions
    Average_JDD = generate_proposal_JDDs(Averaging, Proposal_distributions, number_of_nodes)

    # create list of proposal graphs
    proposals = list(Average_JDD.keys())
    proposal_graphs = {}
    for proposal in proposals:
        proposal_graphs[proposal] = deepcopy(Observered_G)

    # generate best proposal graph reconstructions under specific models
    while proposals:
        for proposal in proposals:
            current_graph = proposal_graphs[proposal]
            current_missing_edges = nx.non_edges(proposal_graphs[proposal])

            # see your score if you add no edges
            total_degrees_1 = max(list(current_graph.degree().values()))
            total_degrees_2 = int(sqrt(len(Average_JDD[proposal].keys())))
            total_degrees = max(total_degrees_1, total_degrees_2)
            JDD_1 = joint_degree_distribution(current_graph, total_degrees+1)
            JDD_2 = Average_JDD[proposal]
            best_eval = norm_difference(JDD_1, JDD_2, 'fro')
            edges_to_add = None

            # now see what happens if we try to add other edges
            for missing_edge in current_missing_edges:
                current_graph.add_edge(missing_edge[0],missing_edge[1])
                total_degrees_1 = max(list(current_graph.degree().values()))
                total_degrees_2 = int(sqrt(len(Average_JDD[proposal].keys())))
                total_degrees = max(total_degrees_1, total_degrees_2)
                JDD_1 = joint_degree_distribution(current_graph,total_degrees)
                JDD_2 = Average_JDD[proposal]
                current_eval = norm_difference(JDD_1, JDD_2, 'fro')
                if best_eval > current_eval:
                    edges_to_add = missing_edge
                    best_eval = current_eval
                current_graph.remove_edge(missing_edge[0], missing_edge[1])

            # now update the current graph proposal and with some probability add random edge
            if edges_to_add is None:
                if uniform() > p:
                    proposals.remove(proposal)
                else:
                    edges_to_add = random.choice(current_missing_edges)
                    proposal_graphs[proposal].add_edge(edges_to_add[0], edges_to_add[1])
            else:
                proposal_graphs[proposal].add_edge(edges_to_add[0],edges_to_add[1])

    # now pick the one that has the lowest norm
    best_eval = 10**20
    best_prediction = None
    best_proposal = None
    for proposal in list(Average_JDD.keys()):
        current_graph = proposal_graphs[proposal]
        total_degrees_1 = max(list(current_graph.degree().values()))
        total_degrees_2 = int(sqrt(len(Average_JDD[proposal].keys())))
        total_degrees = max(total_degrees_1, total_degrees_2)
        JDD_1 = joint_degree_distribution(current_graph,total_degrees)
        JDD_2 = Average_JDD[proposal]
        eval = norm_difference(JDD_1, JDD_2, 'fro')
        #print(proposal,eval)
        if eval < best_eval:
            best_proposal = proposal
            best_eval = eval
            best_prediction = deepcopy(current_graph)

    return best_proposal, best_eval, best_prediction

Proposal_distributions = ['geometric_model','erdos_renyi',
                'random_partition_model','barabasi_albert_model']

# Averaging = 10
# Observered_G = remove_edges(simulate_random_graph('erdos_renyi', 10),.1)
# print()
# print('erdos_renyi')
# print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))
#
# Observered_G = remove_edges(simulate_random_graph('geometric_model', 10),.1)
# print()
# print('geometric_model')
# print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))
#
# Observered_G = remove_edges(simulate_random_graph('random_partition_model', 10),.1)
# print()
# print('random_partition_model')
#
# print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))
# Observered_G = remove_edges(simulate_random_graph('barabasi_albert_model', 10),.1)
# print()
# print('barabasi_albert_model')
# print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))
#

Averaging = 10
percent_removed = linspace(0.01,.99,20)
simulations = 15

for nodes in [10]:
    # for Averaging in [10]:
    #     plots = []
    #     for p in percent_removed:
    #         counter = 0
    #         for i in range(simulations):
    #             remove = 1
    #             while remove:
    #                 G = simulate_random_graph('barabasi_albert_model', nodes)
    #                 Observered_G = remove_edges(G, p)
    #                 if Observered_G:
    #                     remove = 0
    #                     sim = edge_imputation_via_one_step_node_norm_minimization(
    #                     Observered_G, Proposal_distributions, Averaging, 0)
    #                     if sim[0] == 'barabasi_albert_model':
    #                         counter += 1
    #         percent_correct = counter/simulations
    #         plots.append(percent_correct)
    #
    #
    #     plt.plot(percent_removed,plots)
    #     plt.title('Simulation of barabasi_albert_model')
    #     plt.ylabel('percent correct')
    #     plt.xlabel('percent held out')
    #     plt.savefig('barabasi_albert_model' + str(Averaging) + str(nodes))
    #     plt.close()

        plots = []
        for p in percent_removed:
            counter = 0
            for i in range(simulations):
                remove = 1
                while remove:
                    G = simulate_random_graph('erdos_renyi', nodes)
                    Observered_G = remove_edges(G, p)
                    if Observered_G:
                        remove = 0
                        sim = edge_imputation_via_one_step_node_norm_minimization(
                        Observered_G, Proposal_distributions, Averaging, 0)
                        if sim[0] == 'erdos_renyi':
                            counter += 1
            percent_correct = counter/simulations
            plots.append(percent_correct)

        plt.plot(percent_removed,plots)
        plt.title('Simulation of erdos_renyi')
        plt.ylabel('percent correct')
        plt.xlabel('percent held out')
        plt.savefig('erdos_renyi' + str(Averaging) + str(nodes))
        plt.close()

        plots = []
        for p in percent_removed:
            counter = 0
            for i in range(simulations):
                remove = 1
                while remove:
                    G = simulate_random_graph('random_partition_model', nodes)
                    Observered_G = remove_edges(G, p)
                    if Observered_G:
                        remove = 0
                        sim = edge_imputation_via_one_step_node_norm_minimization(
                        Observered_G, Proposal_distributions, Averaging, 0)
                        if sim[0] == 'random_partition_model':
                            counter += 1
            percent_correct = counter/simulations
            plots.append(percent_correct)

        plt.plot(percent_removed,plots)
        plt.title('Simulation of random_partition_model')
        plt.ylabel('percent correct')
        plt.xlabel('percent held out')
        plt.savefig('random_partition_model' + str(Averaging) + str(nodes))
        plt.close()

        plots = []
        for p in percent_removed:
            counter = 0
            for i in range(simulations):
                remove = 1
                while remove:
                    G = simulate_random_graph('geometric_model', nodes)
                    Observered_G = remove_edges(G, p)
                    if Observered_G:
                        remove = 0
                        sim = edge_imputation_via_one_step_node_norm_minimization(
                        Observered_G, Proposal_distributions, Averaging, 0)
                        if sim[0] == 'geometric_model':
                            counter += 1
            percent_correct = counter/simulations
            plots.append(percent_correct)

        plt.plot(percent_removed,plots)
        plt.title('Simulation of geometric_model')
        plt.ylabel('percent correct')
        plt.xlabel('percent held out')
        plt.savefig('geometric_model' + str(Averaging) + str(nodes))
        plt.close()
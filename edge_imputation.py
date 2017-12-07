import networkx as nx
from graph_measures import degree_assortativity, degree_centrality, harmonic_centrality, eigenvector_centrality
from graph_measures import betweenness_centrality, diameter, mean_degree, mean_neighbor_degree
from graph_measures import coefficient_of_variation, global_clustering_coefficient
from random_graphs import random_partition_model,erdos_renyi, geometric_model, barabasi_albert_model
from joint_degree_investigation import joint_degree_distribution
from numpy.linalg import norm
from numpy.random import choice,uniform,randint
from math import floor, sqrt
from random import sample
from copy import deepcopy
import random
from numpy import linspace
from matplotlib import pyplot as plt

"""" Given a graph that is missing data, lets say it has 500 nodes and we think it will have 

    750 (We can do optimal stopping later if we dont know how much we are missing but for now just 

    assume we are missing 33%). Add 5 edges randomly, and do this for x number of trials. Now we 

    have x graphs for 505 nodes each. Pick the best graph out of these x for that approximates each 

    of our generative models. Given we have 5 (Erdos, barabasi, waltz-stroagatz, planted partition, etc), 

    we should end up with 5 "winners". For each of these 5 graphs generate x trials again randomly (add all of 

    these into some pool). Once again pick the best graph out of these x graphs that approximates our distribution.

    In the end we should have 5 graphs each with 750 nodes, and each will be the best approximation 

    of each of our known generative distributions. Now we can simply use this to see which one matches the 

    best with its corresponding one to say we think it is of this type. """

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

def simulate_random_graph(graph_name, number_of_nodes, additional_parameters = None):

    if additional_parameters is not None:
        keys = additional_parameters.keys()

    if graph_name == "random_partition_model":
        if additional_parameters is not None:

            if 'p_in' in keys:
                p_in = additional_parameters['p_in']
            else:
                p_in = uniform()
            if 'p_out' in keys:
                p_out = additional_parameters['p_out']
            else:
                p_out = uniform()
            if 'max_community_sizes' in keys:
                max_community_sizes = additional_parameters['min_community_sizes']
            else:
                max_community_sizes = choice(floor(number_of_nodes / 2))
            if 'min_community_sizes' in keys:
                min_community_sizes = additional_parameters['min_community_sizes']
            else:
                min_community_sizes = 1

            community_sizes = uniform_partition_gen(min_community_sizes,
                                        number_of_nodes, max_community_sizes)
        else:
            p_out = uniform()
            p_in = uniform()
            community_sizes = uniform_partition_gen(1, number_of_nodes, choice(floor(number_of_nodes / 2)))

        return random_partition_model(community_sizes, p_in, p_out)

    elif graph_name == "erdos_renyi":
        if additional_parameters is not None:
            if 'p' in keys:
                p = additional_parameters['p']
            else:
                p = uniform()
        else:
            p = uniform()
        return erdos_renyi(number_of_nodes,p)

    elif graph_name == "geometric_model":
        if additional_parameters is not None:
            if 'r' in keys:
                r = additional_parameters['r']
            else:
                r = floor(sqrt(choice(number_of_nodes)) + 1)
        else:
            r = floor(sqrt(choice(number_of_nodes))+1)
        return geometric_model(number_of_nodes,r)

    elif graph_name == "barabasi_albert_model":
        if additional_parameters is not None:
            if 'c' in keys:
                c = additional_parameters['c']
            else:
                c = choice(floor(number_of_nodes / 2)) + 1
        else:
            c = choice(floor(number_of_nodes/2))+1

        return barabasi_albert_model(number_of_nodes,c)

    else:
        print('please provide a valid graphical model')
        return None

def get_charactoristics(graph_name,n):

    if graph_name == "random_partition_model":

        graph_info = {}
        graph_info['p_in'] = (0,1)
        graph_info['p_out'] = (0, 1)
        graph_info['p_out'] = (0, 1)
        graph_info['min_community_sizes'] = (0, floor(n / 2))
        graph_info['max_community_sizes'] = (0, floor(n / 2))

        return graph_info

    elif graph_name == "erdos_renyi":

        graph_info = {}
        graph_info['p'] = (0,1)

        return graph_info

    elif graph_name == "geometric_model":

        graph_info = {}
        graph_info['r'] = (1,floor(sqrt(n) + 1))

        return graph_info

    elif graph_name == "barabasi_albert_model":

        graph_info = {}
        graph_info['c'] = (1,floor(n / 2) + 1)

        return graph_info

    else:
        print('please provide a valid graphical model')
        return None

def generate_proposal_JDDs(number_of_nodes, distributions, averaging_per_proposal,
                           representations = None, additional_restrictions = None):
    average_JDD = {}
    if (representations is None) and (additional_restrictions is None):
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

    elif representations is not None:

        # if we want to attempt to fit to multiple representations of a random graph, lets try it
        for distribution in distributions:
            info = get_charactoristics(distribution, number_of_nodes)

        for graph in distributions:
            average_JDD[graph] = None
            for sim in range(averaging_per_proposal):
                simulate = 1
                while simulate:
                    current_G = simulate_random_graph(graph, number_of_nodes)
                    if len(list(current_G.degree().values())) > 0:
                        simulate = 0

                total_degrees = max(list(current_G.degree().values()))
                current_JDD = joint_degree_distribution(current_G, total_degrees)

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
        print(proposal,eval)
        if eval < best_eval:
            best_proposal = proposal
            best_eval = eval
            best_prediction = deepcopy(current_graph)

    return best_proposal, best_eval, best_prediction


def remove_edges(g, p):
    edge_set = set()
    edges = g.edges()
    num_edges = g.number_of_edges()

    to_remove = int(floor(num_edges * p))

    removed = 0
    for i in range(to_remove):
        random_edge_index = randint(num_edges)
        edge = edges[random_edge_index]
        edge_set.add((edge[0], edge[1]))
        edge_set.add((edge[1], edge[0]))
        removed += 1

    g.remove_edges_from(list(edge_set))

    return g

# # test 1
# n = 25
# q = 5
# c = 10
# epsilon = 2
# G_1 = get_dd_planted_partition(n, q, c, epsilon)
# G_2 = get_dd_planted_partition(n, q, c, epsilon)
# norm_name = 'fro'
# print(Norm_Difference(G_1,G_2, norm_name))
# simulate_random_graph("random_partition_model", 20).degree()
# print(generate_proposal_JDDs(20, ['random_partition_model'], 10))
# print(generate_proposal_JDDs(20, ['erdos_renyi'], 10))
# print(generate_proposal_JDDs(20, ['geometric_model'], 10))
# print(generate_proposal_JDDs(20, ['barabasi_albert_model'], 10))


if __name__ == "__main__":
    Proposal_distributions = ['geometric_model','erdos_renyi',
                    'random_partition_model','barabasi_albert_model']
    Averaging = 100
    Observered_G = barabasi_albert_model(10,2)
    print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))

    JDD_rp = generate_proposal_JDDs(10, ['random_partition_model'], 1000)
    JDD_er = generate_proposal_JDDs(10, ['erdos_renyi'], 1000)
    JDD_gm = generate_proposal_JDDs(10, ['geometric_model'], 1000)
    JDD_ba = generate_proposal_JDDs(10, ['barabasi_albert_model'], 1000)

    print(norm_difference(JDD_rp,JDD_rp, 'fro'))
    print(norm_difference(JDD_rp,JDD_er, 'fro'))
    print(norm_difference(JDD_rp,JDD_gm, 'fro'))
    print(norm_difference(JDD_rp,JDD_ba, 'fro'))

    print(norm_difference(JDD_er,JDD_er, 'fro'))
    print(norm_difference(JDD_er,JDD_gm, 'fro'))
    print(norm_difference(JDD_er,JDD_ba, 'fro'))

    print(norm_difference(JDD_gm,JDD_gm, 'fro'))
    print(norm_difference(JDD_gm,JDD_ba, 'fro'))

    print(norm_difference(JDD_ba,JDD_ba, 'fro'))

    Proposal_distributions = ['geometric_model','erdos_renyi',
                    'random_partition_model','barabasi_albert_model']
    Averaging = 10
    percent_removed = linspace(0,1,10)
    plots = []
    simulations = 2

    # for p in percent_removed:
    #     counter = 0
    #     for i in range(simulations):
    #         Observered_G = remove_edges(barabasi_albert_model(10, 2), p)
    #         sim = edge_imputation_via_one_step_node_norm_minimization(
    #             Observered_G, Proposal_distributions, Averaging, 0)
    #         if sim[0] == 'barabasi_albert_model':
    #             counter += 1
    #             print(counter)
    #     percent_correct = counter/simulations
    #     plots.append(percent_correct)
    #
    # print(plots)
    # plt.plot(percent_removed,plots)
    # plt.ylabel('Simulation of Barabasi Albert Model')
    # plt.show()


    # Observered_G = remove_edges(barabasi_albert_model(20,2),.01)
    Observered_G = remove_edges(erdos_renyi(10, .999),.9)
    print(edge_imputation_via_one_step_node_norm_minimization(Observered_G, Proposal_distributions, Averaging, 0))

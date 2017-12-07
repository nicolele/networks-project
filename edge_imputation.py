import networkx as nx
from graph_measures import degree_assortativity, degree_centrality, harmonic_centrality, eigenvector_centrality
from graph_measures import betweenness_centrality, diameter, mean_degree, mean_neighbor_degree
from graph_measures import coefficient_of_variation, global_clustering_coefficient

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


def get_dd_planted_partition(n, q, c, epsilon):
    # generate graphs
    p_in = (1 / n) * (c + epsilon / 2)
    p_out = (1 / n) * (c - epsilon / 2)
    nx.planted_partition_graph(int(q), int(n / q), p_in, p_out, seed=42)


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


def generate_distribution_PP(n, q, c, epsilon, statistics_of_interest):
    statistics_distribution_dict = {}

    for stat in statistics_of_interest:
        statistics_distribution_dict[stat] = []

    for i in range(n):
        G = get_dd_planted_partition(n, q, c, epsilon)
        stats = pull_graph_statistics(G)
        for stat in statistics_of_interest:
            new_addition = stats[stat]
            old_list = statistics_distribution_dict[stat]
            old_list.append(new_addition)
            statistics_distribution_dict[stat] = old_list

    return statistics_distribution_dict


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


def find_most_likly_edge_distribution():


# test 1
n = 50;
q = 5;
c = 10;
epsilon = 2
Graph_1 = get_dd_planted_partition(n, q, c, epsilon)

coefficient_of_variation(G)

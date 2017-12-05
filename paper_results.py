"""
This file will attempt to replicate the results given in the paper
"Diversity of graphs with highly variable connectivity"
The main idea of the paper is:
"The purpose of this paper is to explore this notion of
graph diversity and characterize more completely the way in
which the degree sequence of a particular graph dictates
many popular graph features, including its correlation structure"
"""

import random_graphs
from numpy.random import randint
from matplotlib import pyplot as plt
from numpy import std, mean


def graph_assortativity(graph):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for degree_i in graph.degrees():
        sum2 += degree_i**2
        sum3 += .5*degree_i**3
        sum4 += .5*degree_i**2
        for degree_j in graph.degrees():
            sum1 += degree_i*degree_j
    sum2 = sum2**2
    sum4 = sum4**2

    return

def coefficient_of_variation(degree_sequence):
    return std(degree_sequence)/mean(degree_sequence)

def generate_bounds_degree_sequence(ranges,simulations):

    interior_mixing = 10
    min_values = (ranges-2) * [0]
    max_values = (ranges-2) * [0]

    for degree_seq_size in range(2, ranges):

        for sim in range(simulations):

            # pick degree sequence a priori and watch how the s statistic evolves as we add more nodes
            degree_sequence = [randint(1, degree_seq_size) for i in range(0, degree_seq_size)]
            min_value = 10 ** 5
            max_value = 0

            for sim_ in range(interior_mixing):
                if sum(degree_sequence) % 2 == 0:
                    graph = random_graphs.configuration_model(degree_sequence, cleaned=True)
                else:
                    degree_sequence[randint(1, degree_seq_size)] += 1
                    graph = random_graphs.configuration_model(degree_sequence, cleaned=True)
                metric = s_metric(graph)
                if min_value > metric:
                    min_value = metric
                if max_value < metric:
                    max_value = metric

        min_values[degree_seq_size - 2] = min_value
        max_values[degree_seq_size - 2] = max_value

    return min_values, max_values

def get_dd_planted_partition(n, q, c, epsilon):
    n = 50.
    q = 2.  # number of groups
    c = 5.
    epsilon = [0, 4, 8]
    # generate graphs
    p_in = (1 / n) * (c + epsilon[0] / 2)
    p_out = (1 / n) * (c - epsilon[0] / 2)
    nx.planted_partition_graph(int(q), int(n / q), p_in, p_out, seed=42)

def generate_bounds_degree_sequence_planted_partition(ranges,simulations):

    interior_mixing = 10
    min_values = (ranges-2) * [0]
    max_values = (ranges-2) * [0]

    for degree_seq_size in range(2, ranges):

        for sim in range(simulations):

            # pick degree sequence a priori and watch how the s statistic evolves as we add more nodes
            degree_sequence = get_dd_planted_partition()
            min_value = 10 ** 5
            max_value = 0

            for sim_ in range(interior_mixing):
                if sum(degree_sequence) % 2 == 0:
                    graph = random_graphs.configuration_model(degree_sequence, cleaned=True)
                else:
                    degree_sequence[randint(1, degree_seq_size)] += 1
                    graph = random_graphs.configuration_model(degree_sequence, cleaned=True)
                metric = s_metric(graph)
                if min_value > metric:
                    min_value = metric
                if max_value < metric:
                    max_value = metric

        min_values[degree_seq_size - 2] = min_value
        max_values[degree_seq_size - 2] = max_value

    return min_values, max_values


# DEGREE SEQUENCE AND GRAPH DIVERSITY
""" Generate bounds under maximum entropy (Configuration Model) """
ranges = 1000
simulations = 10
mins, maxs = generate_bounds_degree_sequence(ranges, simulations)
print(mins)
print(maxs)
plt.plot(mins)
plt.plot(maxs)
plt.title("Bounds for s metric under the configuration model, approximately uniform degree distribution")
plt.xlabel("number of nodes within graph")
plt.ylabel("minimum and maximum s metric")
plt.show()

"""Next we need to define the coefficient of variation"""
""" Next we are going to verify: For graphs with regular structure that have low variability
in their degree sequence D, there is typically very little diversity
in the corresponding space of graphs G( D). """


""" Look at Coeff for graphs with a degree sequence having an exponential form"""

""" Look at Coeff for graphs that are scale free"""
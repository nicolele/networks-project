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

def s_metric(graph):
    """ the metric measures the extent to which the
    graph g has a hublike core and is maximized when highdegree
    vertices are connected to other high-degree vertices. """
    metric = 0
    for edge in graph.edges():
        node1 = edge[0]
        node2 = edge[1]
        metric += graph.degree(node1) * graph.degree(node2)
    return metric

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

# DEGREE SEQUENCE AND GRAPH DIVERSITY
""" We first consider graphs G(D) that have a particular degree sequence (which might not exist)"""
""" Specifically the configuration model, they derive it analytically, lets look at the empirical bounds
    over random sequences of size n as n->inf"""

#degree_sequence = [randint(1,10) for i in range(1,10)]
#create_configuration_model_random_graph(degree_sequence, cleaned=True)

""" Now we are going to define the s metric, then find the max and minimum values as I increase the 
randomly generated degree sequence"""

ranges = 1000
simulations = 10
mins, maxs = generate_bounds_degree_sequence(ranges, simulations)
print(mins)
print(maxs)
plt.plot(mins)
plt.plot(maxs)
plt.show()

"""Next we need to define the coefficient of variation"""
#degree_sequence = [randint(1,10) for i in range(1,10)]
#print(coefficient_of_variation(degree_sequence))

""" Next we are going to verify: For graphs with regular structure that have low variability
in their degree sequence D, there is typically very little diversity
in the corresponding space of graphs G( D). """


""" Look at Coeff for graphs with a degree sequence having an exponential form"""


""" Look at Coeff for graphs that are scale free"""
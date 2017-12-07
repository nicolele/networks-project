import matplotlib.pyplot as plt
import operator
import numpy as np
import networkx as nx
import random_graphs as rg
from random import shuffle
import graph_measures as gm
from copy import deepcopy
from random_graphs import configuration_model

def joint_degree_distribution(G,total_degrees):

    # returns number of edges connecting nodes of degrees k1 and k2
    joint_distribution = {}
    degrees = list(range(0,total_degrees+1))

    if len(degrees) > 1:
        for degree_1 in degrees:
            for degree_2 in degrees:
                joint_distribution[str(degree_1)+','+str(degree_2)] = 0
                joint_distribution[str(degree_2)+','+str(degree_1)] = 0
        nodes = G.nodes()

        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node_i_degree = G.degree(nodes[i])
                node_j_degree = G.degree(nodes[j])
                joint_distribution[str(node_i_degree) + ',' + str(node_j_degree)] += 1
                joint_distribution[str(node_j_degree) + ',' + str(node_i_degree)] += 1
        return joint_distribution

    else:
        nodes = G.nodes()
        joint_distribution[str(nodes[0]) + ',' + str(nodes[1])] = 1
        joint_distribution[str(nodes[1]) + ',' + str(nodes[0])] = 1
        return joint_distribution

def joint_degree(k_1, k_2, jDD = None, G = None):

    if jDD is None:
        jDD = joint_degree_distribution(G)

    if str(k_1) + ',' + str(k_2) in jDD.keys():
        return jDD[str(k_1) + ',' + str(k_2)]
    elif str(k_2) + ',' + str(k_1) in jDD.keys():
        return jDD[str(k_2) + ',' + str(k_1)]
    else:
        return 0

def feasable_joint_degree_ditributions(proposed_jdd, jdd):

    # now iterate through the combinations to check if any of the values are off
    for key in proposed_jdd.keys():
        if key in jdd:
            return jdd[key] >= proposed_jdd[key]

    return False

def jdd_restricted_configuration_model(degree_sequence, jdd):

    # basic idea is to create bins, and fill the bins as we connect the sticky ends
    degree_vector_ = [[jj for ii in range(0,degree_sequence[jj])] for jj in range(0,len(degree_sequence))]
    degree_vector = [item for sublist in degree_vector_ for item in sublist]
    generative_graph = nx.Graph()

    # deep copy current simulation
    current_simulation = deepcopy(degree_vector)
    shuffle(current_simulation)

    # initializations
    incomplete = 1; counter = 0; edges = []
    max_iterations = 10**10

    # iteratively pull two vertices and then concatinate them to create graph
    while incomplete:
        counter += 1
        # check if start end is not a self loop, or a multi-edge,
        if current_simulation[0] != current_simulation[1] \
        and (current_simulation[0], current_simulation[1]) not in edges \
        and (current_simulation[1], current_simulation[0]) not in edges \
        and current_simulation:

            # then check that we are not not overfilling any of the joint degree ditribution bins
            temp_check = deepcopy(generative_graph)
            temp_check.add_edge(current_simulation[0], current_simulation[1])

            if counter > 3:
                proposed_jdd = joint_degree_distribution(temp_check, max(degree_sequence)+1)
                compatable = feasable_joint_degree_ditributions(proposed_jdd,jdd)

                if compatable:
                    # if so add edge to edge list
                    edges.append((current_simulation[0], current_simulation[1]))
                    generative_graph.add_edge(current_simulation[0], current_simulation[1])
                    # remove first two elements chosen
                    del current_simulation[0]
                    del current_simulation[0]
                    # check if you are out of edges
                    if not current_simulation:
                        incomplete = 0
                        # if not, re-shuffle degree list and try again
                    else:
                        shuffle(current_simulation)
                    if counter > max_iterations:
                        incomplete = 0
            else:
                # if so add edge to edge list
                edges.append((current_simulation[0], current_simulation[1]))
                # remove first two elements chosen
                del current_simulation[0]
                del current_simulation[0]
        else:
            shuffle(current_simulation)
        # check if you are out of edges
        if not current_simulation:
            incomplete = 0
            # if not, re-shuffle degree list and try again
        else:
            shuffle(current_simulation)
        if counter > max_iterations:
            incomplete = 0

    # add simulated graph
    G_ = nx.Graph()
    G_.add_edges_from(edges)

    return G_

def draw_graph(G,title):
    # graph_pos = nx.spring_layout(G)
    graph_pos = nx.spring_layout(G)
    # draw nodes, edges and labels
    nx.draw_networkx_nodes(G, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(G, graph_pos)
    nx.draw_networkx_labels(G, graph_pos, font_size=12, font_family='sans-serif')
    plt.title(title)
    plt.show()

# # now lets test this on a few models
#
# Graph_1 = rg.barabasi_albert_model(10,2)
#
# degree_sequence = sorted(nx.degree(Graph_1).values(),reverse=True)
# joint_degree_dist = joint_degree_distribution(Graph_1, max(degree_sequence)+1)
# print(sum(degree_sequence))
# generated_graph = jdd_restricted_configuration_model(degree_sequence, joint_degree_dist)
#
# print(joint_degree_distribution(generated_graph, max(degree_sequence)+1))
# print(joint_degree_distribution(Graph_1, max(degree_sequence)+1))
#
# draw_graph(Graph_1,'fdf')
#
# draw_graph(generated_graph,'dsfds')




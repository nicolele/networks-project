
from random_graphs import random_partition_model,erdos_renyi, geometric_model, barabasi_albert_model
from numpy.linalg import norm
from numpy.random import choice, uniform
from math import  floor, sqrt
from random import sample
from copy import deepcopy
import numpy as np
from numpy.random import randint
from copy import deepcopy
from numpy import linspace
import networkx as nx
import random

def joint_degree_distribution(G):

    # returns number of edges connecting nodes of degrees k1 and k2
    joint_distribution = {}
    total_degrees = find_max_degree(G)
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

def find_max_degree(G):
    return max(list(G.degree().values()))

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

    random_graphs = []

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
            for p_in_ in p_in:
                for p_out_ in p_out:
                    for max_community_sizes_ in max_community_sizes:
                        for min_community_sizes_ in min_community_sizes:

                            ### check the parameters make sense
                            if max_community_sizes_ < min_community_sizes_:
                                temp = max_community_sizes_
                                max_community_sizes_ = min_community_sizes_
                                min_community_sizes_ = temp

                            ### generate random graph
                            community_sizes = uniform_partition_gen(min_community_sizes_,
                                        number_of_nodes, max_community_sizes_)
                            new_graph = random_partition_model(community_sizes, p_in_, p_out_)
                            if new_graph.nodes():
                                random_graphs.append((new_graph,
                                    {"graph": graph_name, "max_community_sizes": max_community_sizes_,
                                     "min_community_sizes_": min_community_sizes_, "p_in": p_in_,
                                     "p_out": p_out_}))


            return random_graphs

        else:
            p_out = uniform()
            p_in = uniform()
            max_community_sizes = choice(floor(number_of_nodes / 2))
            min_community_sizes = 1
            community_sizes = uniform_partition_gen(min_community_sizes, number_of_nodes, max_community_sizes)
            new_graph = random_partition_model(community_sizes, p_in, p_out)
            if new_graph.nodes():
                random_graphs.append((new_graph,
                {"graph": graph_name, "max_community_sizes": max_community_sizes,
                 "min_community_sizes_": min_community_sizes, "p_in": p_in,
                 "p_out": p_out}))
            else:
                random_graphs = None

            return random_graphs

    elif graph_name == "erdos_renyi":
        if additional_parameters is not None:
            if 'p' in keys:
                p = additional_parameters['p']
                for p_ in p:
                    random_graphs.append((erdos_renyi(number_of_nodes, p_),
                                          {"graph": graph_name, "p": p_}))
            else:
                p = uniform()
                random_graphs.append((erdos_renyi(number_of_nodes, p),
                                      {"graph": graph_name, "p": p}))
        else:
            p = uniform()
            random_graphs.append((erdos_renyi(number_of_nodes,p),
                                {"graph": graph_name, "p": p}))

    elif graph_name == "geometric_model":
        if additional_parameters is not None:
            if 'r' in keys:
                r = additional_parameters['r']
                for r_ in r:
                    random_graphs.append((geometric_model(number_of_nodes,r_),
                                {"graph": graph_name, "r": r_}))
            else:
                r = floor(sqrt(choice(number_of_nodes)) + 1)
                random_graphs.append((geometric_model(number_of_nodes,r),
                                {"graph": graph_name, "r": r}))
        else:
            r = floor(sqrt(choice(number_of_nodes))+1)
            random_graphs.append((geometric_model(number_of_nodes,r),
                                {"graph": graph_name, "r": r}))
        return random_graphs

    elif graph_name == "barabasi_albert_model":

        if additional_parameters is not None:
            if 'c' in keys:
                c = additional_parameters['c']
                for c_ in c:
                    random_graphs.append((barabasi_albert_model(number_of_nodes,c_),
                                         {"graph": graph_name, "c": c_}))
            else:
                c = choice(floor(number_of_nodes / 2)) + 1
                random_graphs.append((barabasi_albert_model(number_of_nodes,c),
                                         {"graph": graph_name, "c": c}))
        else:
            c = choice(floor(number_of_nodes/2))+1
            random_graphs.append((barabasi_albert_model(number_of_nodes,c),
                                         {"graph": graph_name, "c": c}))

        return random_graphs

    else:
        print('please provide a valid graphical model')
        return None

def simulate_random_graphs(graph_names, number_of_nodes, additional_parameters = None):
    proposal_graphs = []
    for graph_name in graph_names:
        new_graph = simulate_random_graph(graph_name, number_of_nodes, additional_parameters)
        if new_graph is not None:
            proposal_graphs.append(new_graph)
    return proposal_graphs

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

def generate_additional_parameters(parameter_mesh_size, number_of_nodes):
    additional_parameters = {}
    additional_parameters['p'] = linspace(0,1,parameter_mesh_size)
    additional_parameters['r'] = linspace(0, sqrt(choice(number_of_nodes)), parameter_mesh_size)
    additional_parameters['c'] = linspace(0, floor(number_of_nodes / 2), parameter_mesh_size)
    additional_parameters['p_in'] = linspace(0, 1, parameter_mesh_size)
    additional_parameters['p_out'] = linspace(0, 1, parameter_mesh_size)
    additional_parameters['max_community_sizes'] = linspace(2, floor(number_of_nodes / 2), parameter_mesh_size)
    additional_parameters['min_community_sizes'] = linspace(2, floor(number_of_nodes / 2), parameter_mesh_size)

def get_rg_params(rg):
    if rg == "random_partition_model":
        return ['p_in', 'p_out', 'min_community_sizes', 'max_community_sizes']
    elif rg == "erdos_renyi":
        return ['p']
    elif rg == "geometric_model":
        return ['r']
    elif rg == "barabasi_albert_model":
        return ['c']
    else:
        return None

def generate_proposal_distributions(number_of_nodes, graph_names, parameter_mesh_size):

    ### generate parameters
    parameters = generate_additional_parameters(parameter_mesh_size, number_of_nodes)

    ### simulate random graphs
    proposal_graphs = simulate_random_graphs(graph_names, number_of_nodes, parameters)

    ### convert them to jdds
    proposal_jdds = []
    for graph in proposal_graphs:
        graph_ = [x for x,_ in graph][0]
        graph_info = [x for _,x in graph][0]
        jdd = joint_degree_distribution(graph_)
        proposal_jdds.append((jdd, graph_info))

    ### return
    return proposal_jdds

def edge_addition(empirical_graph, proposal_jdd, empirical_jdd, max_addition, attempts):

    current_measure = norm_difference(proposal_jdd,empirical_jdd, 'fro')
    possible_edges = nx.non_edges(empirical_graph)
    possible_edges = [edge for edge in possible_edges]
    for attempt in range(attempts):

        ### pick number of edges to add
        edges_to_add = choice(max_addition+1)

        ### pick edges to add
        new_edges = np.random.permutation(possible_edges)[:edges_to_add]

        ### check if the graph got closer
        new_graph = deepcopy(empirical_graph)
        new_graph.add_edges_from(new_edges)
        new_jdd = joint_degree_distribution(new_graph)
        updated_measure = norm_difference(proposal_jdd, new_jdd, 'fro')

        if updated_measure < current_measure:
            return new_graph

    return empirical_graph

def return_closest_match(graph_set, proposal_jdds, averaging):

    best_graph = None
    best_score = 10**20

    for i in range(len(graph_set)):
        for j in range(averaging):
            jdd = joint_degree_distribution(graph_set[i])
            val = norm_difference(jdd, proposal_jdds[i][0], 'fro')
            if val/averaging < best_score:
                best_graph = graph_set[i]

    return best_graph

def find_closest_rg(empirical_graph, rg_graph_names, parameter_mesh_size, max_iterations):

    ### initialization
    number_of_nodes = empirical_graph.number_of_nodes()
    max_addition = number_of_nodes
    attempts = 3

    ### reconstructions
    graph_set = generate_proposal_distributions(number_of_nodes, rg_graph_names, parameter_mesh_size)
    for g in range(len(graph_set)):
        graph_set[g] = empirical_graph

    for iteration in range(max_iterations):

        ### generate new set of proposal distributions
        proposal_jdds = []
        while len(graph_set) != len(proposal_jdds):
            proposal_jdds = generate_proposal_distributions(number_of_nodes, rg_graph_names, parameter_mesh_size)

        ### attempt reconstruction over proposals
        counter = -1
        for jdd in proposal_jdds:
            counter += 1
            current_graph = graph_set[counter]
            empirical_jdd = joint_degree_distribution(current_graph)
            graph_set[counter] = edge_addition(graph_set[counter], jdd[0], empirical_jdd, max_addition, attempts)

    ### return closest match
    averaging = 1
    return return_closest_match(graph_set, proposal_jdds, averaging)

def main():

    ### reconstruction under full graph
    rg_graph_names = ['geometric_model', 'erdos_renyi',
                              'random_partition_model', 'barabasi_albert_model']
    parameter_mesh_size = 2
    max_iterations = 4
    empirical_graph = barabasi_albert_model(10,2)
    print(find_closest_rg(empirical_graph, rg_graph_names, parameter_mesh_size, max_iterations))

    ### reconstruction under 50%
    rg_graph_names = ['geometric_model', 'erdos_renyi',
                              'random_partition_model', 'barabasi_albert_model']
    parameter_mesh_size = 2
    max_iterations = 4
    empirical_graph = barabasi_albert_model(25,2)
    observed_graph = remove_edges(empirical_graph, .1)
    print(find_closest_rg(observed_graph, rg_graph_names, parameter_mesh_size, max_iterations))


if __name__ == "__main__":
    main()
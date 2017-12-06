import networkx as nx
import numpy as np


def process_graph_1(file):
    edge_pairs = []
    with open(file) as f:
        for line in f.readlines():
            line = line.split()
            edge_pairs.append((int(line[0]), int(line[1])))

    g = nx.Graph()
    g.add_edges_from(edge_pairs)
    
    return g


def remove_edges(g, p):
    edge_set = set()
    edges = g.edges()
    num_edges = g.number_of_edges()

    to_remove = np.floor(edges*p)
    
    removed = 0
    while removed <= to_remove:
        random_edge_index = np.random.randint(num_edges)
        edge = edges[random_edge_index]
        edge_set.add((edge[0], edge[1]))
        removed += 1

    g.remove_edges_from(list(edge_set))

    return g


def main():
    g = process_graph_1('data/cit-HepPh.txt')
    remove_edges(g, 0.10)


if __name__ == '__main__':
    main()
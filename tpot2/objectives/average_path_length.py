import networkx as nx
import numpy as np

def average_path_length_objective(graph_pipeline):

    path_lengths =  nx.shortest_path_length(graph_pipeline.graph, source=graph_pipeline.root)
    return np.mean(np.array(list(path_lengths.values())))+1
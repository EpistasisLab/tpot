import networkx as nx
import numpy as np

def average_path_length_objective(graph_pipeline):
    """
    Computes the average shortest path from all nodes to the root/final estimator (only supported for GraphPipeline)

    Parameters
    ----------
    graph_pipeline: GraphPipeline
        The pipeline to compute the average path length for

    """

    path_lengths =  nx.shortest_path_length(graph_pipeline.graph, source=graph_pipeline.root)
    return np.mean(np.array(list(path_lengths.values())))+1
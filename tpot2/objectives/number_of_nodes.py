from ..graphsklearn import GraphPipeline
from sklearn.pipeline import Pipeline

def calculate_graph_number_of_nodes(graphpipeline):
    n_nodes = graphpipeline.graph.number_of_nodes()

    for node in graphpipeline.graph.nodes:
        method = graphpipeline.graph.nodes[node]["instance"]
        if isinstance(method, GraphPipeline):
            n_nodes += calculate_graph_number_of_nodes(method)
            n_nodes -= 1 #don't double count the graph in addition to whats actually in the graph
        if isinstance(method, Pipeline):
            n_nodes += calculate_pipeline_number_of_nodes(method)
            n_nodes -= 1
    
    return n_nodes

def calculate_pipeline_number_of_nodes(pipe):
    n_nodes = len(pipe.named_steps)
    for step in pipe.named_steps:
        method = pipe.named_steps[step]
        if isinstance(method, GraphPipeline):
            n_nodes += calculate_graph_number_of_nodes(method)
            n_nodes -= 1 #don't double count the graph in addition to whats actually in the graph
        if isinstance(method, Pipeline):
            n_nodes += calculate_pipeline_number_of_nodes(method)
            n_nodes -= 1
   
    return n_nodes

def number_of_nodes_objective(est):
    if isinstance(est, GraphPipeline):
        return calculate_graph_number_of_nodes(est)
    if isinstance(est, Pipeline):
        return calculate_pipeline_number_of_nodes(est)
    return 1
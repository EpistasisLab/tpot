
import numpy as np
import tpot2
import networkx as nx
from tpot2.representations.graph_pipeline_individual import GraphIndividual
import random

from tpot2.representations.graph_pipeline_individual.individual import create_node


def estimator_graph_individual_generator(
                                                root_config_dict,
                                                inner_config_dict=None,
                                                leaf_config_dict=None,
                                                max_size = np.inf, 
                                                linear_pipeline = False,
                                                **kwargs,
                                            ) :



        n_nodes = 0
        while True:
            if n_nodes < max_size:
                    n_nodes += 1
            
            for k in root_config_dict.keys():
                
                graph = nx.DiGraph()
                root = create_node(config_dict={k:root_config_dict[k]})
                graph.add_node(root)
                
                ind = tpot2.GraphIndividual(    inner_config_dict=inner_config_dict,  
                                                    leaf_config_dict=leaf_config_dict,
                                                    root_config_dict=root_config_dict,
                                                    initial_graph = graph,

                                                    max_size = max_size, 
                                                    linear_pipeline = linear_pipeline,

                                                    **kwargs,
                                                    )
                
                starting_ops = []
                if inner_config_dict is not None:
                    starting_ops.append(ind._mutate_insert_inner_node)
                if leaf_config_dict is not None:
                    starting_ops.append(ind._mutate_insert_leaf)
                
                if len(starting_ops) > 0:
                    if n_nodes > 0:
                        for _ in range(np.random.randint(0,min(n_nodes,3))):
                            func = np.random.choice(starting_ops)
                            func()

                
                yield ind

            

class BaggingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass

class BoostingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass


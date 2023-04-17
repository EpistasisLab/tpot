
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
                                                max_depth = np.inf,
                                                max_size = np.inf, 
                                                max_children = np.inf,
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
                                                    max_depth = max_depth,
                                                    max_size = max_size, 
                                                    max_children = max_children,
                                                    **kwargs,
                                                    )
                
                
                
                if n_nodes > 0:
                    for _ in range(np.random.randint(0,min(n_nodes,3))):
                        if random.random() < 0.5:
                            ind._mutate_insert_leaf()
                        else: 
                            ind._mutate_insert_inner_node()

                for _ in range(np.random.randint(0,ind.graph.number_of_nodes())):
                     ind._mutate_add_connection_from()
                
                yield ind

            

class BaggingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass

class BoostingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass



import numpy as np
import tpot2
import networkx as nx
from tpot2.individual_representations.graph_pipeline_individual import GraphIndividual

from tpot2.individual_representations.graph_pipeline_individual.individual import create_node


# will randomly generate individuals (no predefined order)
def estimator_graph_individual_generator(
                                                root_config_dict,
                                                inner_config_dict=None,
                                                leaf_config_dict=None,
                                                max_size = np.inf,
                                                linear_pipeline = False,
                                                hyperparameter_probability = 1,
                                                hyper_node_probability = 0,
                                                hyperparameter_alpha = 1,
                                                rng_=None,
                                                **kwargs,
                                            ) :

        rng = np.random.default_rng(rng_)

        while True:

            # if user specified limit, grab a random number between that limit
            if max_size is not np.inf:
                n_nodes = rng.integers(1,max_size+1)
            # else, grab random number between 1,11 (theaksaini)
            else:
                n_nodes = rng.integers(1,11)

            graph = nx.DiGraph()
            root = create_node(config_dict=root_config_dict, rng_=rng) # grab random root model method
            graph.add_node(root)

            ind = GraphIndividual(  rng_=rng,
                                    inner_config_dict=inner_config_dict,
                                    leaf_config_dict=leaf_config_dict,
                                    root_config_dict=root_config_dict,
                                    initial_graph = graph,

                                    max_size = max_size,
                                    linear_pipeline = linear_pipeline,
                                    hyperparameter_probability = hyperparameter_probability,
                                    hyper_node_probability = hyper_node_probability,
                                    hyperparameter_alpha = hyperparameter_alpha,

                                    **kwargs,
                                    )

            starting_ops = []
            if inner_config_dict is not None:
                starting_ops.append(ind._mutate_insert_inner_node)
            if leaf_config_dict is not None:
                starting_ops.append(ind._mutate_insert_leaf)
                n_nodes -= 1

            if len(starting_ops) > 0:
                for _ in range(n_nodes-1):
                    func = rng.choice(starting_ops)
                    func(rng_=rng)

            yield ind


class BaggingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass

class BoostingCompositeGraphSklearn():
    def __init__(self) -> None:
        pass

import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from sklearn.base import BaseEstimator
import sklearn
import networkx as nx
from . import graph_utils
from typing import final

class SklearnIndividual(tpot2.BaseIndividual):

    def __init__(self,) -> None:
        super().__init__()

    def mutate(self, rng=None):
        return
    
    def crossover(self, other, rng=None):
        return

    def export_pipeline(self) -> BaseEstimator:
        return
    
    def unique_id(self):
        return self
    
    @final
    def export_flattened_graphpipeline(self, **graphpipeline_kwargs) -> tpot2.GraphPipeline:
        return flatten_to_graphpipeline(self.export_pipeline(), **graphpipeline_kwargs)

class SklearnIndividualGenerator():
    def __init__(self,):
        pass

    def generate(self, rng=None) -> SklearnIndividual:
        pass






def flatten_graphpipeline(est):
    flattened_full_graph = est.graph.copy()

    #put ests into the node label from the attributes

    flattened_full_graph = nx.relabel_nodes(flattened_full_graph, {n: flattened_full_graph.nodes[n]['instance'] for n in flattened_full_graph.nodes})


    remove_list = []
    for node in flattened_full_graph.nodes:
        if isinstance(node, nx.DiGraph):
            flattened = flatten_any(node)
            
            roots = graph_utils.get_roots(flattened)
            leaves = graph_utils.get_leaves(flattened)

            n1_s = flattened_full_graph.successors(node)
            n1_p = flattened_full_graph.predecessors(node)

            remove_list.append(node)

            flattened_full_graph = nx.compose(flattened_full_graph, flattened)


            flattened_full_graph.add_edges_from([ (n2, n) for n in n1_s for n2 in leaves])
            flattened_full_graph.add_edges_from([ (n, n2) for n in n1_p for n2 in roots])
        
    for node in remove_list:
        flattened_full_graph.remove_node(node)

    return flattened_full_graph

def flatten_pipeline(est):
    graph = nx.DiGraph()
    steps = [flatten_any(s[1]) for s in est.steps]

    #add steps to graph and connect them
    for s in steps:
        graph = nx.compose(graph, s)
    
    #connect leaves of each step to the roots of the next step
    for i in range(len(steps)-1):
        roots = graph_utils.get_roots(steps[i])
        leaves = graph_utils.get_leaves(steps[i+1])
        graph.add_edges_from([ (l,r) for l in leaves for r in roots])
        

    return graph



def flatten_estimator(est):
    graph = nx.DiGraph()
    graph.add_node(est)
    return graph

def flatten_any(est):
    if isinstance(est, tpot2.GraphPipeline):
        return flatten_graphpipeline(est)
    elif isinstance(est, sklearn.pipeline.Pipeline):
        return flatten_pipeline(est)
    else:
        return flatten_estimator(est)


def flatten_to_graphpipeline(est, **graphpipeline_kwargs):
    #rename nodes to string representation of the instance and put the instance in the node attributes
    flattened_full_graph = flatten_any(est)

    instance_to_label = {}
    label_to_instance = {}
    for node in flattened_full_graph.nodes:
        found_unique_label = False
        i=1
        while not found_unique_label:
            new_label = f"{node.__class__.__name__}_{i}"
            if new_label not in label_to_instance:
                found_unique_label = True
            i+=1
        label_to_instance[new_label] = node
        instance_to_label[node] = new_label

    flattened_full_graph = nx.relabel_nodes(flattened_full_graph, instance_to_label)

    for label, instance in label_to_instance.items():
        flattened_full_graph.nodes[label]["instance"] = instance

    return tpot2.GraphPipeline(flattened_full_graph, **graphpipeline_kwargs)
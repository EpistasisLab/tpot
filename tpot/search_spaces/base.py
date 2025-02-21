"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import tpot
import sklearn
from sklearn.base import BaseEstimator
import sklearn
import networkx as nx
from . import graph_utils
from typing import final



class SklearnIndividual(tpot.BaseIndividual):

    def __init_subclass__(cls):
        cls.crossover = cls.validate_same_type(cls.crossover)


    def __init__(self,) -> None:
        super().__init__()

    def mutate(self, rng=None):
        return

    def crossover(self, other, rng=None, **kwargs):
        return 
    
    @final
    def validate_same_type(func):

        def wrapper(self, other, rng=None, **kwargs):
            if not isinstance(other, type(self)):
                return False
            return func(self, other, rng=rng, **kwargs)

        return wrapper

    def export_pipeline(self, **kwargs) -> BaseEstimator:
        return
    
    def unique_id(self):
        """
        Returns a unique identifier for the individual. Used for preventing duplicate individuals from being evaluated.
        """
        return self
    
    #TODO currently TPOT population class manually uses the unique_id to generate the index for the population data frame.
    #alternatively, the index could be the individual itself, with the __eq__ and __hash__ methods implemented.

    # Though this breaks the graphpipeline. When a mutation is called, it changes the __eq__ and __hash__ outputs.
    # Since networkx uses the hash and eq to determine if a node is already in the graph, this causes the graph thing that 
    # This is a new node not in the graph. But this could be changed if when the graphpipeline mutates nodes, 
    # it "replaces" the existing node with the mutated node. This would require a change in the graphpipeline class.

    # def __eq__(self, other):
    #     return self.unique_id() == other.unique_id()
    
    # def __hash__(self):
    #     return hash(self.unique_id())

    #number of components in the pipeline
    def get_size(self):
        return 1
    
    @final
    def export_flattened_graphpipeline(self, **graphpipeline_kwargs) -> tpot.GraphPipeline:
        return flatten_to_graphpipeline(self.export_pipeline(), **graphpipeline_kwargs)

class SearchSpace():
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
    if isinstance(est, tpot.GraphPipeline):
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

    return tpot.GraphPipeline(flattened_full_graph, **graphpipeline_kwargs)
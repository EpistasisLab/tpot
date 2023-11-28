import numpy as np
from tpot2 import config
import networkx as nx
from abc import abstractmethod
import matplotlib.pyplot as plt
import sklearn
import tpot2
import sklearn.pipeline
from typing import Generator
import optuna
from itertools import combinations
from .graph_utils import graph_utils
import itertools
import baikal
import copy
from .. import BaseIndividual

class NodeLabel():
    def __init__(self, *,

        #intialized, but may change later
            method_class = None, #transformer or baseestimator
            hyperparameters=None,
            label=None,
    ):

        #intializable, but may change later
        self.method_class = method_class #transformer or baseestimator
        self.hyperparameters = hyperparameters
        self.label = label
        self._params = None



from functools import partial
#@https://stackoverflow.com/questions/20530455/isomorphic-comparison-of-networkx-graph-objects-instead-of-the-default-address

class GraphKey():
    '''
    A class that can be used as a key for a graph.

    Parameters
    ----------
    graph : (nx.Graph)
        The graph to use as a key. Node Attributes are used for the hash.
    matched_label : (str)
        The node attribute to consider for the hash.
    '''

    def __init__(self, graph, matched_label='label') -> None:#['hyperparameters', 'method_class']) -> None:


        self.graph = graph
        self.matched_label = matched_label
        self.node_match = partial(node_match, matched_labels=[matched_label])
        self.key = int(nx.weisfeiler_lehman_graph_hash(self.graph, node_attr=self.matched_label),16) #hash(tuple(sorted([val for (node, val) in self.graph.degree()])))


    #If hash is different, node is definitely different
    # https://arxiv.org/pdf/2002.06653.pdf
    def __hash__(self) -> int:

        return self.key

    #If hash is same, use __eq__ to know if they are actually different
    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, node_match=self.node_match)

def node_match(n1,n2, matched_labels):
    return all( [ n1[m] == n2[m] for m in matched_labels])


class GraphIndividual(BaseIndividual):
    '''
    An individual that contains a template for a graph sklearn pipeline.

    Parameters
    ----------
    root_config_dict : {dict with format {method class: param_function}}
        A dictionary of methods and functions that return a dictionary of hyperparameters.
        Used to create the root node of the graph.
    inner_config_dict : {dict with format {method class: param_function}}
        A dictionary of methods and functions that return a dictionary of hyperparameters.
        Used to create the inner nodes of the graph. If None, uses root_config_dict.
    leaf_config_dict : {dict with format {method class: param_function}}
        A dictionary of methods and functions that return a dictionary of hyperparameters.
        Used to create the leaf nodes of the graph. If not None, then all leafs must be created from this dictionary.
        Otherwise leaves will be created from inner_config_dict.
    initial_graph : (nx.DiGraph or list):
        A graph to initialize the individual with.
        If a list, it will initialize a linear graph with the methods in the list in the sequence provided.
        If the items in the list are dictionaries, nodes will be itialized with those dictionaries.
        Strings in the list correspond to the default configuration files. They can be 'Selector', 'Regressor', 'Transformer', 'Classifier'.
    max_depth : (int)
        The maximum depth of the graph as measured by the shortest distance from the root.
    max_size : (int)
        The maximum number of nodes in the graph.
    max_children : (int)
        The maximum number of children a node can have.
    name : (str)
        The name of the individual.
    crossover_same_depth : (bool)
        If true, then crossover will only occur between nodes of the same depth as measured by the shortest distance from the root.
    crossover_same_recursive_depth : (bool)
        If the graph is recursive, then crossover will only occur between graphs of the same recursive depth as measured by the shortest distance from the root.
    '''
    def __init__(
                self,
                root_config_dict,
                inner_config_dict=None,
                leaf_config_dict=None,
                initial_graph = None,
                max_size = np.inf,
                linear_pipeline = False,
                name=None,
                crossover_same_depth = False,
                crossover_same_recursive_depth = True,

                hyperparameter_probability = 1,
                hyper_node_probability = 0,
                hyperparameter_alpha = 1,

                unique_subset_values = None,
                initial_subset_values = None,
                rng_=None,
                ):

        self.__debug = False

        rng = np.random.default_rng(rng_)

        self.root_config_dict = root_config_dict
        self.inner_config_dict = inner_config_dict
        self.leaf_config_dict = leaf_config_dict


        self.max_size = max_size
        self.name = name

        self.crossover_same_depth = crossover_same_depth
        self.crossover_same_recursive_depth = crossover_same_recursive_depth

        self.unique_subset_values = unique_subset_values
        self.initial_subset_values = initial_subset_values

        self.hyperparameter_probability = hyperparameter_probability
        self.hyper_node_probability = hyper_node_probability
        self.hyperparameter_alpha = hyperparameter_alpha

        if self.unique_subset_values is not None:
            self.row_subset_selector = tpot2.representations.SubsetSelector(rng_=rng, values=unique_subset_values, initial_set=initial_subset_values,k=20)

        if isinstance(initial_graph, nx.DiGraph):
            self.graph = initial_graph
            self.root = list(nx.topological_sort(self.graph))[0]

            if self.leaf_config_dict is not None and len(self.graph.nodes) == 1:
                first_leaf = create_node(self.leaf_config_dict, rng_=rng)
                self.graph.add_edge(self.root,first_leaf)

        elif isinstance(initial_graph, list):
            node_list = []
            for item in initial_graph:
                if isinstance(item, dict):
                    node_list.append(create_node(item, rng_=rng))
                elif isinstance(item, str):
                    if item == 'Selector':
                            from tpot2.config import selector_config_dictionary
                            node_list.append(create_node(selector_config_dictionary, rng_=rng))
                    elif  item == 'Regressor':
                            from tpot2.config import regressor_config_dictionary
                            node_list.append(create_node(regressor_config_dictionary, rng_=rng))
                    elif  item == 'Transformer':
                            from tpot2.config import transformer_config_dictionary
                            node_list.append(create_node(transformer_config_dictionary, rng_=rng))
                    elif  item == 'Classifier':
                            from tpot2.config import classifier_config_dictionary
                            node_list.append(create_node(classifier_config_dictionary, rng_=rng))

            self.graph = nx.DiGraph()
            for child, parent in zip(node_list, node_list[1:]):
                self.graph.add_edge(parent, child)

            self.root = node_list[-1]

        else:
            self.graph = nx.DiGraph()

            self.root = create_node(self.root_config_dict, rng_=rng)
            self.graph.add_node(self.root)

            if self.leaf_config_dict is not None:
                first_leaf = create_node(self.leaf_config_dict, rng_=rng)
                self.graph.add_edge(self.root,first_leaf)



        self.initialize_all_nodes(rng_=rng)

        #self.root =list(nx.topological_sort(self.graph))[0]


        self.mutate_methods_list =     [self._mutate_hyperparameters,
                                        self._mutate_replace_node,
                                        self._mutate_remove_node,
                                        ]

        self.crossover_methods_list = [
                                        self._crossover_swap_branch,
                                        ]


        if self.inner_config_dict is not None:
            self.mutate_methods_list.append(self._mutate_insert_inner_node)
            self.crossover_methods_list.append(self._crossover_take_branch) #this is the only crossover method that can create inner nodes
            if not linear_pipeline:
                self.mutate_methods_list.append(self._mutate_insert_bypass_node)
                self.mutate_methods_list.append(self._mutate_remove_edge)
                self.mutate_methods_list.append(self._mutate_add_edge)

        if not linear_pipeline and (self.leaf_config_dict is not None or self.inner_config_dict is not None):
            self.mutate_methods_list.append(self._mutate_insert_leaf)




        if self.unique_subset_values is not None:
            self.crossover_methods_list.append(self._crossover_row_subsets)
            self.mutate_methods_list.append(self._mutate_row_subsets )

        self.optimize_methods_list = [ #self._optimize_optuna_single_method_full_pipeline,
                                        self._optimize_optuna_all_methods_full_pipeline]

        self.key = None

    def select_config_dict(self, node):
        #check if the node is root, leaf, or inner
        if len(list(self.graph.predecessors(node))) == 0: #root
            return self.root_config_dict
        elif self.leaf_config_dict is not None and len(list(self.graph.successors(node))) == 0: #leaf
            return self.leaf_config_dict
        else: #inner
            return self.inner_config_dict


    def initialize_all_nodes(self, rng_=None):
        rng = np.random.default_rng(rng_)
        for node in self.graph:
            if isinstance(node,GraphIndividual):
                continue
            if node.method_class is None:
                node.method_class = rng.choice(list(self.select_config_dict(node).keys()))
            if node.hyperparameters is None:
                get_hyperparameter(self.select_config_dict(node)[node.method_class], nodelabel=node,  alpha=self.hyperparameter_alpha, hyperparameter_probability=self.hyperparameter_probability)


    def fix_noncompliant_leafs(self, rng_=None):
        rng = np.random.default_rng(rng_)
        leafs = [node for node in self.graph.nodes if len(list(self.graph.successors(node)))==0]
        compliant_leafs = []
        noncompliant_leafs = []
        for leaf in leafs:
            if leaf.method_class in self.leaf_config_dict:
                compliant_leafs.append(leaf)
            else:
                noncompliant_leafs.append(leaf)

        #find all good leafs. If no good leaves exist, create a new one
        if len(compliant_leafs) == 0:
            first_leaf = NodeLabel(config_dict=self.leaf_config_dict)
            first_leaf.method_class = rng.choice(list(first_leaf.config_dict.keys())) #TODO: check when there is no new method
            first_leaf.hyperparameters = first_leaf.config_dict[first_leaf.method_class](config.hyperparametersuggestor)
            get_hyperparameter(self.select_config_dict(first_leaf)[first_leaf.method_class], nodelabel=first_leaf,  alpha=self.hyperparameter_alpha, hyperparameter_probability=self.hyperparameter_probability)
            compliant_leafs.append(first_leaf)

        #connect bad leaves to good leaves (making them internal nodes)
        if len(noncompliant_leafs) > 0:
            for node in noncompliant_leafs:
                self.graph.add_edge(node, rng.choice(compliant_leafs))




    def _merge_duplicated_nodes(self):

        graph_changed = False
        merged = False
        while(not merged):
            node_list = list(self.graph.nodes)
            merged = True
            for node, other_node in itertools.product(node_list, node_list):
                if node is other_node or isinstance(node,GraphIndividual) or isinstance(other_node,GraphIndividual):
                    continue

                #If nodes are same class/hyperparameters
                if node.method_class == other_node.method_class and node.hyperparameters == other_node.hyperparameters:
                    node_children = set(self.graph.successors(node))
                    other_node_children = set(self.graph.successors(other_node))
                    #if nodes have identical children, they can be merged
                    if node_children == other_node_children:
                        for other_node_parent in list(self.graph.predecessors(other_node)):
                            if other_node_parent not in self.graph.predecessors(node):
                                self.graph.add_edge(other_node_parent,node)

                        self.graph.remove_node(other_node)
                        merged=False
                        graph_changed = True
                        break

        return graph_changed

    #returns a flattened pipeline
    def flatten_pipeline(self,depth=0):
        flattened_full_graph = self.graph.copy()
        remove_list = []
        for node in flattened_full_graph:
            if isinstance(node,GraphIndividual):
                flattened = node.flatten_pipeline(depth+1)
                roots = graph_utils.get_roots(flattened)
                leaves = graph_utils.get_leaves(flattened)

                n1_s = flattened_full_graph.successors(node)
                n1_p = flattened_full_graph.predecessors(node)

                remove_list.append(node)


                flattened_full_graph = nx.compose(flattened_full_graph, flattened)


                flattened_full_graph.add_edges_from([ (n2, n) for n in n1_s for n2 in leaves])
                flattened_full_graph.add_edges_from([ (n, n2) for n in n1_p for n2 in roots])
            else:
                flattened_full_graph.nodes[node]['recursive depth'] = depth


        for node in remove_list:
            flattened_full_graph.remove_node(node)

        if self.unique_subset_values is not None:
            for node in flattened_full_graph:
                if "subset_values" not in flattened_full_graph.nodes[node]:
                    flattened_full_graph.nodes[node]["subset_values"] = list(self.row_subset_selector.subsets)
                else:
                    #intersection
                    flattened_full_graph.nodes[node]["subset_values"] = list(set(flattened_full_graph.nodes[node]["subset_values"]) & set(self.row_subset_selector.subsets))

        return flattened_full_graph

    def get_num_nodes(self,):
        num_nodes = 0

        for node in self.graph.nodes:
            if isinstance(node, GraphIndividual):
                num_nodes+= node.get_num_nodes()
            else:
                num_nodes += 1

        return num_nodes


    def export_nested_pipeline(self, **graph_pipeline_args):

        flattened_full_graph = self.graph.copy()
        remove_list = []
        for node in list(flattened_full_graph.nodes):
            if isinstance(node,GraphIndividual):
                gp = node.export_pipeline(**graph_pipeline_args)

                n1_s = flattened_full_graph.successors(node)
                n1_p = flattened_full_graph.predecessors(node)

                remove_list.append(node)

                flattened_full_graph.add_node(gp)


                flattened_full_graph.add_edges_from([ (gp, n) for n in n1_s])
                flattened_full_graph.add_edges_from([ (n, gp) for n in n1_p])


        for node in remove_list:
            flattened_full_graph.remove_node(node)

        estimator_graph = flattened_full_graph

        #mapping = {node:node.method_class(**node.hyperparameters) for node in estimator_graph}
        label_remapping = {}
        label_to_instance = {}

        for node in estimator_graph:
            found_unique_label = False
            i=1
            while not found_unique_label:
                print(type(node))
                if type(node) is tpot2.GraphPipeline:
                    label = "GraphPipeline_{0}".format( i)
                else:
                    label = "{0}_{1}".format(node.method_class.__name__, i)
                if label not in label_to_instance:
                    found_unique_label = True
                else:
                    i+=1


            if type(node) is tpot2.GraphPipeline:
                label_remapping[node] = label
                label_to_instance[label] = node
            else:
                label_remapping[node] = label
                label_to_instance[label] = node.method_class(**node.hyperparameters)

        estimator_graph = nx.relabel_nodes(estimator_graph, label_remapping)

        for label, instance in label_to_instance.items():
            estimator_graph.nodes[label]["instance"] = instance

        return tpot2.GraphPipeline(graph=estimator_graph, **graph_pipeline_args)

    def export_pipeline(self, **graph_pipeline_args):
        estimator_graph = self.flatten_pipeline()

        #mapping = {node:node.method_class(**node.hyperparameters) for node in estimator_graph}
        label_remapping = {}
        label_to_instance = {}

        for node in estimator_graph:
            found_unique_label = False
            i=1
            while not found_unique_label:
                label = "{0}_{1}".format(node.method_class.__name__, i)
                if label not in label_to_instance:
                    found_unique_label = True
                else:
                    i+=1

            label_remapping[node] = label
            label_to_instance[label] = node.method_class(**node.hyperparameters)

        estimator_graph = nx.relabel_nodes(estimator_graph, label_remapping)

        for label, instance in label_to_instance.items():
            estimator_graph.nodes[label]["instance"] = instance

        return tpot2.GraphPipeline(graph=estimator_graph, **graph_pipeline_args)

    def export_baikal(self,):
        graph = self.flatten_pipeline()
        toposorted = list(nx.topological_sort(graph))
        toposorted.reverse()
        node_outputs = {}

        X = baikal.Input('X')
        y = baikal.Input('Target')

        for i in range(len(toposorted)):
            node = toposorted[i]
            if len(list(graph.successors(node))) == 0: #If this node had no inputs use X
                this_inputs = X
            else: #in node has inputs, get those
                this_inputs = [node_outputs[child] for child in graph.successors(node)]

            this_output = baikal.make_step(node.method_class, class_name=node.method_class.__name__)(**node.hyperparameters)(this_inputs,y)
            node_outputs[node] = this_output

            if i == len(toposorted)-1: #last method doesn't need transformed.
                return baikal.Model(inputs=X, outputs=this_output, targets=y)


    def plot(self):
        G = self.flatten_pipeline().reverse() #self.graph.reverse()
        #TODO clean this up
        try:
            pos = nx.planar_layout(G)  # positions for all nodes
        except:
            pos = nx.shell_layout(G)
        # nodes
        options = {'edgecolors': 'tab:gray', 'node_size': 800, 'alpha': 0.9}
        nodelist = list(G.nodes)
        node_color = [plt.cm.Set1(G.nodes[n]['recursive depth']) for n in G]

        fig, ax = plt.subplots()

        nx.draw(G, pos, nodelist=nodelist, node_color=node_color, ax=ax,  **options)


        '''edgelist = []
        for n in n1.node_set:
            for child in n.children:
                edgelist.append((n,child))'''

        # edges
        #nx.draw_networkx_edges(G, pos, width=3.0, arrows=True)
        '''nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edgelist],
            width=8,
            alpha=0.5,
            edge_color='tab:red',
        )'''



        # some math labels
        labels = {}
        for i, n in enumerate(G.nodes):
            labels[n] = n.method_class.__name__ + "\n" + str(n.hyperparameters)


        nx.draw_networkx_labels(G, pos, labels,ax=ax, font_size=7, font_color='black')

        plt.tight_layout()
        plt.axis('off')
        plt.show()


    #############

    #TODO currently does not correctly return false when adding a leaf causes a duplicate node that is later merged
    def mutate(self, rng_=None):
        rng = np.random.default_rng(rng_)
        self.key = None
        graph = self.select_graphindividual(rng_=rng)
        return graph._mutate(rng_=rng)

    def _mutate(self, rng_=None):
        rng = np.random.default_rng(rng_)
        rng.shuffle(self.mutate_methods_list)
        for mutate_method in self.mutate_methods_list:
            if mutate_method(rng_=rng):
                self._merge_duplicated_nodes()

                if self.__debug:
                    print(mutate_method)

                    if self.root not in self.graph.nodes:
                        print('lost root something went wrong with ', mutate_method)

                    if len(self.graph.predecessors(self.root)) > 0:
                        print('root has parents ', mutate_method)

                    if any([n in nx.ancestors(self.graph,n) for n in self.graph.nodes]):
                        print('a node is connecting to itself...')

                    if self.__debug:
                        try:
                            nx.find_cycle(self.graph)
                            print('something went wrong with ', mutate_method)
                        except:
                            pass

                return True

        return False

    def _mutate_row_subsets(self, rng_=None):
        rng = np.random.default_rng(rng_)
        if self.unique_subset_values is not None:
            self.row_subset_selector.mutate(rng_=rng)


    def _mutate_hyperparameters(self, rng_=None):
        '''
        Mutates the hyperparameters for a randomly chosen node in the graph.
        '''
        rng = np.random.default_rng(rng_)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        completed_one = False
        for node in sorted_nodes_list:
            if isinstance(node,GraphIndividual):
                continue
            if isinstance(self.select_config_dict(node)[node.method_class], dict):
                continue

            if not completed_one:
                _,_, completed_one = get_hyperparameter(self.select_config_dict(node)[node.method_class], rng_=rng, nodelabel=node,  alpha=self.hyperparameter_alpha, hyperparameter_probability=self.hyperparameter_probability)
            else:
                if self.hyper_node_probability > rng.random():
                    get_hyperparameter(self.select_config_dict(node)[node.method_class], rng_=rng, nodelabel=node,  alpha=self.hyperparameter_alpha, hyperparameter_probability=self.hyperparameter_probability)

        return completed_one




    def _mutate_replace_node(self, rng_=None):
        '''
        Replaces the method in a randomly chosen node by a method from the available methods for that node.

        '''
        rng = np.random.default_rng(rng_)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        for node in sorted_nodes_list:
            if isinstance(node,GraphIndividual):
                continue
            new_node = create_node(self.select_config_dict(node), rng_=rng)
            #check if new node and old node are the same
            #TODO: add attempts?
            if node.method_class != new_node.method_class or node.hyperparameters != new_node.hyperparameters:
                nx.relabel_nodes(self.graph, {new_node:node}, copy=False)
                return True

        return False


    def _mutate_remove_node(self, rng_=None):
        '''
        Removes a randomly chosen node and connects its parents to its children.
        If the node is the only leaf for an inner node and 'leaf_config_dict' is not none, we do not remove it.
        '''
        rng = np.random.default_rng(rng_)
        nodes_list = list(self.graph.nodes)
        nodes_list.remove(self.root)
        leaves = graph_utils.get_leaves(self.graph)

        while len(nodes_list) > 0:
            node = rng.choice(nodes_list)
            nodes_list.remove(node)

            if self.leaf_config_dict is not None and len(list(nx.descendants(self.graph,node))) == 0 : #if the node is a leaf
                if len(leaves) <= 1:
                    continue #dont remove the last leaf
                leaf_parents = self.graph.predecessors(node)

                # if any of the parents of the node has one one child, continue
                if any([len(list(self.graph.successors(lp))) < 2 for lp in leaf_parents]): #dont remove a leaf if it is the only input into another node.
                    continue

                graph_utils.remove_and_stitch(self.graph, node)
                graph_utils.remove_nodes_disconnected_from_node(self.graph, self.root)
                return True

            else:
                graph_utils.remove_and_stitch(self.graph, node)
                graph_utils.remove_nodes_disconnected_from_node(self.graph, self.root)
                return True

        return False

    def _mutate_remove_edge(self, rng_=None):
        '''
        Deletes an edge as long as deleting that edge does not make the graph disconnected.
        '''
        rng = np.random.default_rng(rng_)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        for child_node in sorted_nodes_list:
            parents = list(self.graph.predecessors(child_node))
            if len(parents) > 1: # if it has more than one parent, you can remove an edge (if this is the only child of a node, it will become a leaf)

                for parent_node in parents:
                    # if removing the egde will make the parent_node a leaf node, skip
                    if self.leaf_config_dict is not None and len(list(self.graph.successors(parent_node))) < 2:
                        continue

                    self.graph.remove_edge(parent_node, child_node)
                    return True
        return False

    def _mutate_add_edge(self, rng_=None):
        '''
        Randomly add an edge from a node to another node that is not an ancestor of the first node.
        '''
        rng = np.random.default_rng(rng_)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        for child_node in sorted_nodes_list:
            for parent_node in sorted_nodes_list:
                if self.leaf_config_dict is not None:
                    if len(list(self.graph.successors(parent_node))) == 0:
                        continue

                # skip if
                # - parent and child are the same node
                # - edge already exists
                # - child is an ancestor of parent
                if  (child_node is not parent_node) and not self.graph.has_edge(parent_node,child_node) and (child_node not in nx.ancestors(self.graph, parent_node)):
                    self.graph.add_edge(parent_node,child_node)
                    return True

        return False


    def _mutate_insert_leaf(self, rng_=None):
        rng = np.random.default_rng(rng_)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            for node in sorted_nodes_list:
                #if leafs are protected, check if node is a leaf
                #if node is a leaf, skip because we don't want to add node on top of node
                if (self.leaf_config_dict is not None #if leafs are protected
                    and   len(list(self.graph.successors(node))) == 0 #if node is leaf
                    and  len(list(self.graph.predecessors(node))) > 0 #except if node is root, in which case we want to add a leaf even if it happens to be a leaf too
                    ):

                    continue

                #If node *is* the root or is not a leaf, add leaf node. (dont want to add leaf on top of leaf)
                if self.leaf_config_dict is not None:
                    new_node = create_node(self.leaf_config_dict, rng_=rng)
                else:
                    new_node = create_node(self.inner_config_dict, rng_=rng)

                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                return True

        return False

    def _mutate_insert_bypass_node(self, rng_=None):
        rng = np.random.default_rng(rng_)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            rng.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                for child_node in sorted_nodes_list2:
                    if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                        if self.leaf_config_dict is not None:
                            #If if we are protecting leafs, dont add connection into a leaf
                            if len(list(nx.descendants(self.graph,node))) ==0 :
                                continue

                            new_node = create_node(config_dict = self.inner_config_dict, rng_=rng)

                            self.graph.add_node(new_node)
                            self.graph.add_edges_from([(node, new_node), (new_node, child_node)])
                            return True

        return False


    def _mutate_insert_inner_node(self, rng_=None):
        rng = np.random.default_rng(rng_)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            rng.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                #loop through children of node
                for child_node in list(self.graph.successors(node)):

                    if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                        if self.leaf_config_dict is not None:
                            #If if we are protecting leafs, dont add connection into a leaf
                            if len(list(nx.descendants(self.graph,node))) ==0 :
                                continue

                            new_node = create_node(config_dict = self.inner_config_dict, rng_=rng)

                            self.graph.add_node(new_node)
                            self.graph.add_edges_from([(node, new_node), (new_node, child_node)])
                            self.graph.remove_edge(node, child_node)
                            return True

        return False

    ######################################################
    # Crossover

    def get_graphs(self):
        graphs = [self]
        self.graph.graph['depth'] = 0
        self.graph.graph['recursive depth'] = 0
        for node in self.graph.nodes:
            if isinstance(node, GraphIndividual):
                node.graph.graph['depth'] = nx.shortest_path_length(self.graph, self.root, node)
                graphs = graphs + node._get_graphs(depth=1)

        return graphs


    def _get_graphs(self, depth=1):
        graphs = [self]
        self.graph.graph['recursive depth'] = depth
        for node in self.graph.nodes:
            if isinstance(node, GraphIndividual):
                node.graph.graph['depth'] = nx.shortest_path_length(self.graph, self.root, node)
                graphs = graphs + node._get_graphs(depth=depth+1)

        return graphs


    def select_graphindividual(self, rng_=None):
        rng = np.random.default_rng(rng_)
        graphs = self.get_graphs()
        weights = [g.graph.number_of_nodes() for g in graphs]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights] # generate probabilities based on sum of weights
        return rng.choice(graphs, p=weights)


    def select_graph_same_recursive_depth(self,ind1,ind2,rng_=None):
        rng = np.random.default_rng(rng_)

        graphs1 = ind1.get_graphs()
        weights1 = [g.graph.number_of_nodes() for g in graphs1]
        w1_sum = sum(weights1)
        weights1 = [w / w1_sum for w in weights1]

        graphs2 = ind2.get_graphs()
        weights2 = [g.graph.number_of_nodes() for g in graphs2]
        w2_sum = sum(weights2)
        weights2 = [w / w2_sum for w in weights2]

        g1_sorted_graphs = random_weighted_sort(graphs1, weights1, rng)
        g2_sorted_graphs = random_weighted_sort(graphs2, weights2, rng)

        for g1, g2 in zip(g1_sorted_graphs, g2_sorted_graphs):
            if g1.graph.graph['depth'] == g2.graph.graph['depth'] and g1.graph.graph['recursive depth'] == g2.graph.graph['recursive depth']:
                return g1, g2

        return ind1,ind2

    def crossover(self, ind2, rng_=None):
        '''
        self is the first individual, ind2 is the second individual
        If crossover_same_depth, it will select graphindividuals at the same recursive depth.
        Otherwise, it will select graphindividuals randomly from the entire graph and its subgraphs.

        This does not impact graphs without subgraphs. And it does not impacts nodes that are not graphindividuals. Cros
        '''

        rng = np.random.default_rng(rng_)

        self.key = None
        ind2.key = None
        if self.crossover_same_recursive_depth:
            # selects graphs from the same recursive depth and same depth from the root
            g1, g2 = self.select_graph_same_recursive_depth(self, ind2, rng_=rng)


        else:
            g1 = self.select_graphindividual(rng_=rng)
            g2 = ind2.select_graphindividual(rng_=rng)

        return g1._crossover(g2, rng_=rng)

    def _crossover(self, Graph, rng_=None):
        rng = np.random.default_rng(rng_)

        rng.shuffle(self.crossover_methods_list)
        for crossover_method in self.crossover_methods_list:
            if crossover_method(Graph, rng_=rng):
                self._merge_duplicated_nodes()
                return True

        if self.__debug:
            try:
                nx.find_cycle(self.graph)
                print('something went wrong with ', crossover_method)
            except:
                pass

        return False


    def _crossover_row_subsets(self, G2, rng_=None):
        rng = np.random.default_rng(rng_)
        if self.unique_subset_values is not None and G2.unique_subset_values is not None:
            self.row_subset_selector.crossover(G2.row_subset_selector, rng_=rng)


    def _crossover_swap_node(self, G2, rng_=None):
        '''
        Swaps randomly chosen node from Parent1 with a randomly chosen node from Parent2.
        '''
        rng = np.random.default_rng(rng_)

        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng_=rng)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph, rng_=rng)

        for node1, node2 in pair_gen:
            if not (node1 is self.root or node2 is G2.root): #TODO: allow root

                n1_s = self.graph.successors(node1)
                n1_p = self.graph.predecessors(node1)

                n2_s = G2.graph.successors(node2)
                n2_p = G2.graph.predecessors(node2)

                self.graph.remove_node(node1)
                G2.graph.remove_node(node2)

                self.graph.add_node(node2)

                self.graph.add_edges_from([ (node2, n) for n in n1_s])
                G2.graph.add_edges_from([ (node1, n) for n in n2_s])

                self.graph.add_edges_from([ (n, node2) for n in n1_p])
                G2.graph.add_edges_from([ (n, node1) for n in n2_p])

                return True
        return False



    def _crossover_swap_branch(self, G2, rng_=None):
        '''
        swaps a branch from parent1 with a branch from parent2. does not modify parent2
        '''
        rng = np.random.default_rng(rng_)

        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng_=rng)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph, rng_=rng)

        for node1, node2 in pair_gen:
            #TODO: if root is in inner_config_dict, then do use it?
            if node1 is self.root or node2 is G2.root: #dont want to add root as inner node
                continue

            #check if node1 is a leaf and leafs are protected, don't add an input to the leave
            if self.leaf_config_dict is not None: #if we are protecting leaves,
                node1_is_leaf = len(list(self.graph.successors(node1))) == 0
                node2_is_leaf = len(list(G2.graph.successors(node2))) == 0
                #if not ((node1_is_leaf and node1_is_leaf) or (not node1_is_leaf and not node2_is_leaf)): #if node1 is a leaf
                if (node1_is_leaf and (not node2_is_leaf)) or ( (not node1_is_leaf) and node2_is_leaf):
                    #only continue if node1 and node2 are both leaves or both not leaves
                    continue

            temp_graph_1 = self.graph.copy()
            temp_graph_1.remove_node(node1)
            graph_utils.remove_nodes_disconnected_from_node(temp_graph_1, self.root)

            #isolating the branch
            branch2 = G2.graph.copy()
            n2_descendants = nx.descendants(branch2,node2)
            for n in list(branch2.nodes):
                if n not in n2_descendants and n is not node2: #removes all nodes not in the branch
                    branch2.remove_node(n)

            branch2 = copy.deepcopy(branch2)
            branch2_root = graph_utils.get_roots(branch2)[0]
            temp_graph_1.add_edges_from(branch2.edges)
            for p in list(self.graph.predecessors(node1)):
                temp_graph_1.add_edge(p,branch2_root)

            if temp_graph_1.number_of_nodes() > self.max_size:
                continue

            self.graph = temp_graph_1

            return True
        return False

    #TODO: Currently returns true even if hyperparameters are blank
    def _crossover_hyperparameters(self, G2, rng_=None):
        '''
        Swaps the hyperparamters of one randomly chosen node in Parent1 with the hyperparameters of randnomly chosen node in Parent2.
        '''
        rng = np.random.default_rng(rng_)

        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng_=rng)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph, rng_=rng)

        for node1, node2 in pair_gen:
            if isinstance(node1,GraphIndividual) or isinstance(node2,GraphIndividual):
                continue

            if node1.method_class == node2.method_class:
                tmp = node1.hyperparameters
                node1.hyperparameters = node2.hyperparameters
                node2.hyperparameters = tmp
                return True

        return False

    #not including the nodes, just their children
    #Finds leaves attached to nodes and swaps them
    def _crossover_swap_leaf_at_node(self, G2, rng_=None):
        rng = np.random.default_rng(rng_)

        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng_=rng)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph, rng_=rng)

        success = False
        for node1, node2 in pair_gen:
            # if leaves are protected node1 and node2 must both be leaves or both be inner nodes
            if self.leaf_config_dict is not None and not (len(list(self.graph.successors(node1)))==0 ^ len(list(G2.graph.successors(node2)))==0):
                continue
            #self_leafs = [c for c in nx.descendants(self.graph,node1) if len(list(self.graph.successors(c)))==0 and c is not node1]
            node_leafs = [c for c in nx.descendants(G2.graph,node2) if len(list(G2.graph.successors(c)))==0 and c is not node2]

            # if len(self_leafs) >0:
            #     for c in self_leafs:
            #         if random.choice([True,False]):
            #             self.graph.remove_node(c)
            #             G2.graph.add_edge(node2, c)
            #             success = True

            if len(node_leafs) >0:
                for c in node_leafs:
                    if rng.choice([True,False]):
                        G2.graph.remove_node(c)
                        self.graph.add_edge(node1, c)
                        success = True

        return success


    def _crossover_take_branch(self, G2, rng_=None):
        '''
        Takes a subgraph from Parent2 and add it to a randomly chosen node in Parent1.
        '''
        rng = np.random.default_rng(rng_)

        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng_=rng)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph, rng_=rng)

        for node1, node2 in pair_gen:
            #TODO: if root is in inner_config_dict, then do use it?
            if node2 is G2.root: #dont want to add root as inner node
                continue


            #check if node1 is a leaf and leafs are protected, don't add an input to the leave
            if self.leaf_config_dict is not None and len(list(self.graph.successors(node1))) == 0:
                continue

            #icheck if node2 is graph individual
            # if isinstance(node2,GraphIndividual):
            #     if not ((isinstance(node2,GraphIndividual) and ("Recursive" in self.inner_config_dict or "Recursive" in self.leaf_config_dict))):
            #         continue

            #isolating the branch
            branch2 = G2.graph.copy()
            n2_descendants = nx.descendants(branch2,node2)
            for n in list(branch2.nodes):
                if n not in n2_descendants and n is not node2: #removes all nodes not in the branch
                    branch2.remove_node(n)

            #if node1 plus node2 branch has more than max_children, skip
            if branch2.number_of_nodes() + self.graph.number_of_nodes() > self.max_size:
                continue

            branch2 = copy.deepcopy(branch2)
            branch2_root = graph_utils.get_roots(branch2)[0]
            self.graph.add_edges_from(branch2.edges)
            self.graph.add_edge(node1,branch2_root)

            return True
        return False

    #TODO: swap all leaf nodes
    def _crossover_swap_all_leafs(self, G2, rng_=None):
        pass


    #TODO: currently ignores ensembles, make it include nodes inside of ensembles
    def optimize(self, rng_, objective_function, steps=5):
        rng = np.random.default_rng(rng_)
        rng.shuffle(self.optimize_methods_list) #select an optimization method
        for optimize_method in self.optimize_methods_list:
            if optimize_method(rng, objective_function, steps=steps):
                return True

    #optimize the hyperparameters of one method to improve the entire pipeline
    def _optimize_optuna_single_method_full_pipeline(self, rng_, objective_function, steps=5):
        rng = np.random.default_rng(rng_)
        nodes_list = list(self.graph.nodes)
        rng.shuffle(nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
        for node in nodes_list:
            if not isinstance(node, NodeLabel) or isinstance(self.select_config_dict(node)[node.method_class],dict):
                continue
            else:
                study = optuna.create_study()

                def objective(trial):
                    params = self.select_config_dict(node)[node.method_class](trial)
                    node.hyperparameters = params

                    trial.set_user_attr('params', params)
                    try:
                        return objective_function(self)
                    except:
                        return np.NAN

                study.optimize(objective, n_trials=steps)
                node.hyperparameters = study.best_trial.user_attrs['params']
                return True


    #optimize the hyperparameters of all methods simultaneously to improve the entire pipeline
    def _optimize_optuna_all_methods_full_pipeline(self, rng_, objective_function, steps=5):
        nodes_list = list(self.graph.nodes)
        study = optuna.create_study()
        nodes_to_optimize = []
        for node in nodes_list:
            if not isinstance(node, NodeLabel) or isinstance(self.select_config_dict(node)[node.method_class],dict):
                continue
            else:
                nodes_to_optimize.append(node)

        def objective(trial):
            param_list = []
            for i, node in enumerate(nodes_to_optimize):
                params = self.select_config_dict(node)[node.method_class](trial, name=f'node_{i}')
                node.hyperparameters = params
                param_list.append(params)

            trial.set_user_attr('params', param_list)

            try:
                return objective_function(self)
            except:
                return np.NAN

        study.optimize(objective, n_trials=steps)
        best_params = study.best_trial.user_attrs['params']

        for node, params in zip(nodes_to_optimize,best_params):
            node.hyperparameters = params

        return True


    def _cached_transform(cache_nunber=0):
        #use a cache for models at each CV fold?
        #cache just transformations at each fold?
        #TODO how to separate full model?
        pass

    def __str__(self):
        return self.export_pipeline().__str__()

    def unique_id(self) -> GraphKey:
        if self.key is None:
            g = self.flatten_pipeline()
            for n in g.nodes:
                if "subset_values" in g.nodes[n]:
                    g.nodes[n]['label'] = {n.method_class: n.hyperparameters, "subset_values":g.nodes[n]["subset_values"]}
                else:
                    g.nodes[n]['label'] = {n.method_class: n.hyperparameters}

                g.nodes[n]['method_class'] = n.method_class #TODO making this transformation doesn't feel very clean?
                g.nodes[n]['hyperparameters'] = n.hyperparameters

            g = nx.convert_node_labels_to_integers(g)
            self.key = GraphKey(graph=g)

        return self.key

    def full_node_list(self):
        node_list = list(self.graph.nodes)
        for node in node_list:
            if isinstance(node, GraphIndividual):
                node_list.pop(node_list.index(node))
                node_list.extend(node.graph.nodes)
        return node_list




def create_node(config_dict, rng_=None):
    '''
    Takes a config_dict and returns a node with a random method_class and hyperparameters
    '''
    rng = np.random.default_rng(rng_)
    method_class = rng.choice(list(config_dict.keys()))
    #if method_class == GraphIndividual or method_class == 'Recursive':
    if method_class == 'Recursive':
        node = GraphIndividual(**config_dict[method_class])
    else:
        hyperparameters, params, _ = get_hyperparameter(config_dict[method_class], rng_=rng, nodelabel=None)

        node = NodeLabel(
                                        method_class=method_class,
                                        hyperparameters=hyperparameters
                                        )
        node._params = params

    return node


def random_weighted_sort(l,weights, rng_=None):
    rng = np.random.default_rng(rng_)
    sorted_l = []
    indeces = {i: weights[i] for i in range(len(l))}
    while len(indeces) > 0:
        keys = list(indeces.keys())
        p = np.array([indeces[k] for k in keys])
        p = p / p.sum()
        next_item = rng.choice(list(indeces.keys()), p=p)
        indeces.pop(next_item)
        sorted_l.append(l[next_item])

    return sorted_l


def get_hyperparameter(config_func, rng_, nodelabel=None,  alpha=1, hyperparameter_probability=1):
    rng = np.random.default_rng(rng_)
    changed = False
    if isinstance(config_func, dict):
        return config_func, None, changed

    if nodelabel is not None:
        trial = config.hyperparametersuggestor.Trial(rng_=rng, old_params=nodelabel._params, alpha=alpha, hyperparameter_probability=hyperparameter_probability)
        new_params = config_func(trial)
        changed = trial._params != nodelabel._params
        nodelabel._params = trial._params
        nodelabel.hyperparameters = new_params
    else:
        trial = config.hyperparametersuggestor.Trial(rng_=rng, old_params=None, alpha=alpha, hyperparameter_probability=hyperparameter_probability)
        new_params = config_func(trial)

    return  new_params, trial._params, changed
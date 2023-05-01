import numpy as np
import random
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

class NodeLabel():
    def __init__(self, *,

        #intialized, but may change later
            method_class = None, #transformer or baseestimator
            hyperparameters=None,
    ):

        #intializable, but may change later
        self.method_class = method_class #transformer or baseestimator
        self.hyperparameters = hyperparameters

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


class GraphIndividual(tpot2.BaseIndividual):
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
                max_depth = np.inf,
                max_size = np.inf, 
                max_children = np.inf,
                name=None,
                crossover_same_depth = False,
                crossover_same_recursive_depth = True,
                

                unique_subset_values = None,
                initial_subset_values = None,
                ):

        self.__debug = False

        self.root_config_dict = root_config_dict
        if inner_config_dict is None:
            self.inner_config_dict = self.root_config_dict
        else:
            self.inner_config_dict = inner_config_dict
        self.leaf_config_dict = leaf_config_dict

        self.max_depth = max_depth
        self.max_size = max_size
        self.name = name
        self.max_children = max_children
        self.crossover_same_depth = crossover_same_depth
        self.crossover_same_recursive_depth = crossover_same_recursive_depth

        self.unique_subset_values = unique_subset_values
        self.initial_subset_values = initial_subset_values

        if self.unique_subset_values is not None:
            self.row_subset_selector = tpot2.representations.SubsetSelector(values=unique_subset_values, initial_set=initial_subset_values,k=20)

        if isinstance(initial_graph, nx.DiGraph):
            self.graph = initial_graph
            self.root = list(nx.topological_sort(self.graph))[0]

            if self.leaf_config_dict is not None and len(self.graph.nodes) == 1:
                first_leaf = create_node(self.leaf_config_dict)
                self.graph.add_edge(self.root,first_leaf)

        elif isinstance(initial_graph, list):
            node_list = []
            for item in initial_graph:
                if isinstance(item, dict):
                    node_list.append(create_node(item))
                elif isinstance(item, str):
                    if item == 'Selector':
                            from tpot2.config import selector_config_dictionary
                            node_list.append(create_node(selector_config_dictionary))
                    elif  item == 'Regressor':
                            from tpot2.config import regressor_config_dictionary
                            node_list.append(create_node(regressor_config_dictionary))
                    elif  item == 'Transformer':
                            from tpot2.config import transformer_config_dictionary
                            node_list.append(create_node(transformer_config_dictionary))
                    elif  item == 'Classifier': 
                            from tpot2.config import classifier_config_dictionary
                            node_list.append(create_node(classifier_config_dictionary))
        
            self.graph = nx.DiGraph()
            for child, parent in zip(node_list, node_list[1:]):
                self.graph.add_edge(parent, child)

            self.root = node_list[-1]

        else:
            self.graph = nx.DiGraph()
            
            self.root = create_node(self.root_config_dict)
            self.graph.add_node(self.root)

            if self.leaf_config_dict is not None:
                first_leaf = create_node(self.leaf_config_dict)
                self.graph.add_edge(self.root,first_leaf)


 
        self.initialize_all_nodes()

        #self.root =list(nx.topological_sort(self.graph))[0]


        self.mutate_methods_list = [    self._mutate_hyperparameters,
                                        self._mutate_replace_node, 
                                        self._mutate_insert_leaf,
                                        self._mutate_remove_node,

                                        ]



        self.crossover_methods_list = [
                                    #self._crossover_swap_node,
                                    #self._crossover_hyperparameters,
                                    self._crossover_swap_branch,
                                    self._crossover_take_branch,
                                    #self._crossover_swap_leaf_at_node,
                                            ]

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


    def initialize_all_nodes(self,):
        for node in self.graph:
            if isinstance(node,GraphIndividual):
                continue
            if node.method_class is None:
                node.method_class = random.choice(list(self.select_config_dict(node).keys()))
            if node.hyperparameters is None:
                node.hyperparameters = self.select_config_dict(node)[node.method_class](config.hyperparametersuggestor)

    def fix_noncompliant_leafs(self):
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
            first_leaf.method_class = random.choice(list(first_leaf.config_dict.keys())) #TODO: check when there is no new method
            first_leaf.hyperparameters = first_leaf.config_dict[first_leaf.method_class](config.hyperparametersuggestor)
            compliant_leafs.append(first_leaf)

        #connect bad leaves to good leaves (making them internal nodes)
        if len(noncompliant_leafs) > 0:
            for node in noncompliant_leafs:
                self.graph.add_edge(node, random.choice(compliant_leafs))

    # max_depth = np.inf,
    # max_size = np.inf, 
    # max_children = np.inf,

    # TENTATIVE: do we need the following function?
    def prune_to_limits(self,):
        


        #Find all leaves that are too deep,
        # Get the sequence for the longest path
        # remove nodes from path, starting from the leaves 


        # nodelist = graph_utils.get_leaves(self.graph)
        # if not np.isinf(self.max_depth):
        #     for node in nodelist:
        #         done = False
        #         while not done: #TODO make more efficient
        #             done = True
        #             max_path = graph_utils.get_max_path_size(self.graph, self.root, node, return_path=True)
        #             if len(max_path) > self.max_depth:
        #                 max_path.reverse()
        #                 if self.leaf_config_dict is not None:
        #                     max_path.remove(node)
        #                 number_to_remove = len(max_path) - self.max_depth
        #                 #max_path.remove(node)
        #                 for i in range(number_to_remove):
        #                     graph_utils.remove_and_stitch(self.graph, max_path[i])
        #                     done = False
                        

            
        #max_size
        nodelist = list(self.graph.nodes)
        nodelist.remove(self.root)
        if len(nodelist) > self.max_size:
            random.shuffle(nodelist)
            for n in nodelist[0:len(nodelist) - self.max_size]:
                graph_utils.remove_and_stitch(self.graph, n)

        #max children
        nodelist= list(self.graph.nodes)
        for node in nodelist:
            #if a node has more children than allowed
            successors = list(self.graph.successors(node))
            num_successors = len(successors)
            if num_successors > self.max_children:
                #move the extra children
                random.shuffle(successors)
                for c in successors[0:num_successors-self.max_children]:
                    self.graph.remove_edge(node,c)

        graph_utils.remove_nodes_disconnected_from_node(self.graph, self.root)


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
    def mutate(self,):
        self.key = None
        graph = self.select_graphindividual()
        return graph._mutate()

    def _mutate(self,):
        random.shuffle(self.mutate_methods_list)
        for mutate_method in self.mutate_methods_list:
            if mutate_method():
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

    def _mutate_row_subsets(self,):
        if self.unique_subset_values is not None:
            self.row_subset_selector.mutate()


    def _mutate_hyperparameters(self):
        '''
        Mutates the hyperparameters for a randomly chosen node in the graph.
        '''
        sorted_nodes_list = list(self.graph.nodes)
        random.shuffle(sorted_nodes_list) 
        for node in sorted_nodes_list:
            if isinstance(node,GraphIndividual):
                continue
            if isinstance(self.select_config_dict(node)[node.method_class], dict):
                continue
            node.hyperparameters = self.select_config_dict(node)[node.method_class](config.hyperparametersuggestor) 
            
            return True
        return False

    def _mutate_replace_node(self):
        '''
        Replaces the method in a randomly chosen node by a method from the available methods for that node.

        '''
        sorted_nodes_list = list(self.graph.nodes)
        random.shuffle(sorted_nodes_list) 
        for node in sorted_nodes_list:
            if isinstance(node,GraphIndividual):
                continue
            node.method_class = random.choice(list(self.select_config_dict(node).keys())) 
            if isinstance(self.select_config_dict(node)[node.method_class], dict):
                hyperparameters = self.select_config_dict(node)[node.method_class]
            else: 
                hyperparameters = self.select_config_dict(node)[node.method_class](config.hyperparametersuggestor)
            node.hyperparameters = hyperparameters
            return True
            
        return False


    def _mutate_remove_node(self):
        '''
        Removes a randomly chosen node and connects its parents to its children.
        If the node is the only leaf for an inner node and 'leaf_config_dict' is not none, we do not remove it.
        '''
        nodes_list = list(self.graph.nodes)
        nodes_list.remove(self.root)
        leaves = graph_utils.get_leaves(self.graph)

        while len(nodes_list) > 0:
            node = random.choices(nodes_list,)[0]
            nodes_list.remove(node)

            if self.leaf_config_dict is not None and len(list(nx.descendants(self.graph,node))) == 0 : #if the node is a leaf
                if len(leaves) <= 1:
                    continue #dont remove the last leaf
                leaf_parents = self.graph.predecessors(node)

                # if any of the parents of the node has one one child, continue
                if any([len(list(nx.descendants(self.graph,lp))) < 2 for lp in leaf_parents]): #dont remove a leaf if it is the only input into another node.
                    continue

            else:
                graph_utils.remove_and_stitch(self.graph, node)
            
            return True
        return False

    def _mutate_remove_edge(self):
        '''
        Deletes an edge as long as deleting that edge does not make the graph disconnected.
        '''
        sorted_nodes_list = list(self.graph.nodes)
        random.shuffle(sorted_nodes_list) 
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

    def _mutate_add_edge(self):
        '''
        Randomly add an edge from a node to another node that is not an ancestor of the first node.
        '''
        sorted_nodes_list = list(self.graph.nodes)
        random.shuffle(sorted_nodes_list)
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


    def _mutate_insert_leaf(self):
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            random.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            for node in sorted_nodes_list:
                #if leafs are protected, check if node is a leaf
                #if node is a leaf, skip because we don't want to add node on top of node
                if (self.leaf_config_dict is not None #if leafs are protected
                    and   len(list(self.graph.successors(node))) == 0 #if node is leaf
                    and  len(list(self.graph.predecessors(node))) > 0 #except if node is root, in which case we want to add a leaf even if it happens to be a leaf too
                    ):
                    
                    
                    continue

                if self.max_children > len(list(self.graph.successors(node))):
                    if np.isinf(self.max_depth) or  self.max_depth >= 1+  graph_utils.get_max_path_size(self.graph, self.root, node): #stackoverflow, check, can it be more efficient?:
                        

                        
                            #If node *is* the root or is not a leaf, add leaf node. (dont want to add leaf on top of leaf)
                        if self.leaf_config_dict is not None:
                            new_node = create_node(self.leaf_config_dict)
                        else:
                            new_node = create_node(self.inner_config_dict)


                        self.graph.add_node(new_node)
                        self.graph.add_edge(node, new_node)
                        return True

        return False

    def _mutate_insert_bypass_node(self):
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            random.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            random.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                if self.max_children > len(self.graph):
                    for child_node in sorted_nodes_list2:
                        if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                            if self.leaf_config_dict is not None:
                                #If if we are protecting leafs, dont add connection into a leaf
                                if len(list(nx.descendants(self.graph,node))) ==0 :
                                    continue
                            
                            #If adding this node will not make the graph too deep
                            if np.isinf(self.max_depth) or  self.max_depth > graph_utils.get_max_path_through_node(self.graph, self.root, child_node): #this is pretty inneficient.

                                    new_node = create_node(config_dict = self.inner_config_dict)

                                    self.graph.add_node(new_node)
                                    self.graph.add_edges_from([(node, new_node), (new_node, child_node)])
                                    return True

        return False


    def _mutate_insert_inner_node(self):
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            random.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            random.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                if self.max_children > len(self.graph):
                    
                    #loop through children of node
                    for child_node in list(self.graph.successors(node)):
                        
                        
                        if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                            if self.leaf_config_dict is not None:
                                #If if we are protecting leafs, dont add connection into a leaf
                                if len(list(nx.descendants(self.graph,node))) ==0 :
                                    continue
                            
                            #If adding this node will not make the graph too deep
                            if np.isinf(self.max_depth) or  self.max_depth > graph_utils.get_max_path_through_node(self.graph, self.root, child_node): #this is pretty inneficient.

                                    new_node = create_node(config_dict = self.inner_config_dict)

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


    def select_graphindividual(self,):
        graphs = self.get_graphs()
        weights = [g.graph.number_of_nodes() for g in graphs]
        return random.choices(graphs, weights=weights)[0]

    def select_graph_same_recursive_depth(self,ind1,ind2):
        graphs1 = ind1.get_graphs()
        weights1 = [g.graph.number_of_nodes() for g in graphs1]
        graphs2 = ind2.get_graphs()
        weights2 = [g.graph.number_of_nodes() for g in graphs2]
        
        g1_sorted_graphs = random_weighted_sort(graphs1, weights1)
        g2_sorted_graphs = random_weighted_sort(graphs2, weights2)

        for g1, g2 in zip(g1_sorted_graphs, g2_sorted_graphs):
            if g1.graph.graph['depth'] == g2.graph.graph['depth'] and g1.graph.graph['recursive depth'] == g2.graph.graph['recursive depth']:
                return g1, g2

        return ind1,ind2

    def crossover(self, ind2):
        '''
        self is the first individual, ind2 is the second individual
        If crossover_same_depth, it will select graphindividuals at the same recursive depth.
        Otherwise, it will select graphindividuals randomly from the entire graph and its subgraphs.

        This does not impact graphs without subgraphs. And it does not impacts nodes that are not graphindividuals. Cros
        '''
  
        self.key = None
        ind2.key = None
        if self.crossover_same_recursive_depth:
            # selects graphs from the same recursive depth and same depth from the root
            g1, g2 = self.select_graph_same_recursive_depth(self, ind2)
            
            
        else:
            g1 = self.select_graphindividual()
            g2 = ind2.select_graphindividual()

        return g1._crossover(g2)
    
    def _crossover(self, Graph):
    
        random.shuffle(self.crossover_methods_list)
        for crossover_method in self.crossover_methods_list:
            if crossover_method(Graph):
                self._merge_duplicated_nodes()

            self._merge_duplicated_nodes()

            return True

        if self.__debug:
            try:
                nx.find_cycle(self.graph)
                print('something went wrong with ', crossover_method)
            except: 
                pass

        return False


    def _crossover_row_subsets(self, G2):
        if self.unique_subset_values is not None and G2.unique_subset_values is not None:
            self.row_subset_selector.crossover(G2.row_subset_selector)
    

    def _crossover_swap_node(self, G2):
        '''
        Swaps randomly chosen node from Parent1 with a randomly chosen node from Parent2.
        '''
        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph)

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



    def _crossover_swap_branch(self, G2):
        '''
        Swaps a subgraph from Parent1 with a subgraph from Parent2.
        '''
        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph)

        for node1, node2 in pair_gen:
            if node1 is self.root or node2 is G2.root:
                continue

            # check if node2 type is a graph individual
            if isinstance(node2,GraphIndividual):
                if not ((isinstance(node2,GraphIndividual) and ("Recursive" in self.inner_config_dict or "Recursive" in self.leaf_config_dict))):
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

            temp_graph_1.add_edges_from(branch2.edges)
            for p in list(self.graph.predecessors(node1)):
                temp_graph_1.add_edge(p,node2)


            if temp_graph_1.number_of_nodes() > self.max_size:
                continue

            self.graph = temp_graph_1
            return True

        return False

    #TODO: Currently returns true even if hyperparameters are blank
    def _crossover_hyperparameters(self, G2):
        '''
        Swaps the hyperparamters of one randomly chosen node in Parent1 with the hyperparameters of randnomly chosen node in Parent2.
        '''
        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph)

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
    def _crossover_swap_leaf_at_node(self, G2):
        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph)

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
                    if random.choice([True,False]):
                        G2.graph.remove_node(c)
                        self.graph.add_edge(node1, c)
                        success = True

        return success


    def _crossover_take_branch(self, G2):
        '''
        Takes a subgraph from Parent2 and add it to a randomly chosen node in Parent1.
        '''
        if self.crossover_same_depth:
            pair_gen = graph_utils.select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root)
        else:
            pair_gen = graph_utils.select_nodes_randomly(self.graph, G2.graph)

        for node1, node2 in pair_gen:
            if node1 is self.root or node2 is G2.root:
                continue

            #icheck if node2 is graph individual
            if isinstance(node2,GraphIndividual):
                if not ((isinstance(node2,GraphIndividual) and ("Recursive" in self.inner_config_dict or "Recursive" in self.leaf_config_dict))):
                    continue

            #isolating the branch
            branch2 = G2.graph.copy()
            n2_descendants = nx.descendants(branch2,node2)
            for n in list(branch2.nodes):
                if n not in n2_descendants and n is not node2: #removes all nodes not in the branch
                    branch2.remove_node(n)

            #if node1 plus node2 branch has more than max_children, skip
            if branch2.number_of_nodes() + self.graph.number_of_nodes() > self.max_size:
                continue

            self.graph.add_edges_from(branch2.edges)
            self.graph.add_edge(node1,node2)

            return True
        return False

    #TODO: swap all leaf nodes
    def _crossover_swap_all_leafs(self, G2):
        pass


    #TODO: currently ignores ensembles, make it include nodes inside of ensembles
    def optimize(self, objective_function, steps=5):
        random.shuffle(self.optimize_methods_list) #select an optimization method
        for optimize_method in self.optimize_methods_list:
            if optimize_method(objective_function, steps=steps):
                return True

    #optimize the hyperparameters of one method to improve the entire pipeline
    def _optimize_optuna_single_method_full_pipeline(self, objective_function, steps=5):
        nodes_list = list(self.graph.nodes)
        random.shuffle(nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
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
    def _optimize_optuna_all_methods_full_pipeline(self, objective_function, steps=5):
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





def create_node(config_dict):
    '''
    Takes a config_dict and returns a node with a random method_class and hyperparameters
    '''
    method_class = random.choice(list(config_dict.keys()))
    #if method_class == GraphIndividual or method_class == 'Recursive':
    if method_class == 'Recursive':
        node = GraphIndividual(**config_dict[method_class])
    else:
        if isinstance(config_dict[method_class], dict):
            hyperparameters = config_dict[method_class]
        else: 
            hyperparameters = config_dict[method_class](config.hyperparametersuggestor)

        node = NodeLabel(
                                        method_class=method_class,
                                        hyperparameters=hyperparameters)
    return node




import random
def random_weighted_sort(l,weights):
    sorted_l = []
    indeces = {i: weights[i] for i in range(len(l))}
    while len(indeces) > 0:
        next_item = random.choices(list(indeces.keys()), weights=list(indeces.values()))[0]
        indeces.pop(next_item)
        sorted_l.append(l[next_item])
    
    return sorted_l
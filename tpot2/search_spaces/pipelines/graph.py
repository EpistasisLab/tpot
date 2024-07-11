import tpot2
import numpy as np
from typing import Generator, List, Tuple, Union
from ..base import SklearnIndividual, SklearnIndividualGenerator
import networkx as nx
import copy
import matplotlib.pyplot as plt
import itertools
from ..graph_utils import *
from ..nodes.estimator_node import EstimatorNodeIndividual
from typing import Union, Callable
import sklearn
from functools import partial
import random

class GraphPipelineIndividual(SklearnIndividual):
    """
        Defines a search space of pipelines in the shape of a Directed Acyclic Graphs. The search spaces for root, leaf, and inner nodes can be defined separately if desired.
        Each graph will have a single root serving as the final estimator which is drawn from the `root_search_space`. If the `leaf_search_space` is defined, all leaves 
        in the pipeline will be drawn from that search space. If the `leaf_search_space` is not defined, all leaves will be drawn from the `inner_search_space`.
        Nodes that are not leaves or roots will be drawn from the `inner_search_space`. If the `inner_search_space` is not defined, there will be no inner nodes.

        `cross_val_predict_cv`, `method`, `memory`, and `use_label_encoder` are passed to the GraphPipeline object when the pipeline is exported and not directly used in the search space.

        Exports to a GraphPipeline object.

        Parameters
        ----------

        root_search_space: SklearnIndividualGenerator
            The search space for the root node of the graph. This node will be the final estimator in the pipeline.
        
        inner_search_space: SklearnIndividualGenerator, optional
            The search space for the inner nodes of the graph. If not defined, there will be no inner nodes.
        
        leaf_search_space: SklearnIndividualGenerator, optional
            The search space for the leaf nodes of the graph. If not defined, the leaf nodes will be drawn from the inner_search_space.
            
        crossover_same_depth: bool, optional
            If True, crossover will only occur between nodes at the same depth in the graph. If False, crossover will occur between nodes at any depth.
        
        cross_val_predict_cv: int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy used in inner classifiers or regressors

        method: str, optional
            The prediction method to use for the inner classifiers or regressors. If 'auto', it will try to use predict_proba, decision_function, or predict in that order.

        memory: str or object with the joblib.Memory interface, optional
            Used to cache the input and outputs of nodes to prevent refitting or computationally heavy transformations. By default, no caching is performed. If a string is given, it is the path to the caching directory.

        use_label_encoder: bool, optional
            If True, the label encoder is used to encode the labels to be 0 to N. If False, the label encoder is not used.
            Mainly useful for classifiers (XGBoost) that require labels to be ints from 0 to N.
            Can also be a sklearn.preprocessing.LabelEncoder object. If so, that label encoder is used.
        
        rng: int, RandomState instance or None, optional
            Seed for sampling the first graph instance. 
            
        """
    
    def __init__(
            self,  
            root_search_space: SklearnIndividualGenerator, 
            leaf_search_space: SklearnIndividualGenerator = None, 
            inner_search_space: SklearnIndividualGenerator = None, 
            max_size: int = np.inf,
            crossover_same_depth: bool = False,
            cross_val_predict_cv: Union[int, Callable] = 0, #signature function(estimator, X, y=none)
            method: str = 'auto',
            memory=None,
            use_label_encoder: bool = False,
            rng=None):
        
        super().__init__()

        self.__debug = False

        rng = np.random.default_rng(rng)

        self.root_search_space = root_search_space
        self.leaf_search_space = leaf_search_space
        self.inner_search_space = inner_search_space
        self.max_size = max_size
        self.crossover_same_depth = crossover_same_depth

        self.cross_val_predict_cv = cross_val_predict_cv
        self.method = method
        self.memory = memory
        self.use_label_encoder = use_label_encoder

        self.root = self.root_search_space.generate(rng)
        self.graph = nx.DiGraph()
        self.graph.add_node(self.root)

        if self.leaf_search_space is not None:
            self.leaf = self.leaf_search_space.generate(rng)
            self.graph.add_node(self.leaf)
            self.graph.add_edge(self.root, self.leaf)

        if self.inner_search_space is None and self.leaf_search_space is None:
            self.mutate_methods_list = [self._mutate_node]
            self.crossover_methods_list = [self._crossover_swap_branch,]#[self._crossover_swap_branch, self._crossover_swap_node, self._crossover_take_branch]  #TODO self._crossover_nodes, 

        else:
            self.mutate_methods_list = [self._mutate_insert_leaf, self._mutate_insert_inner_node, self._mutate_remove_node, self._mutate_node, self._mutate_insert_bypass_node]
            self.crossover_methods_list = [self._crossover_swap_branch, self._crossover_nodes, self._crossover_take_branch ]#[self._crossover_swap_branch, self._crossover_swap_node, self._crossover_take_branch]  #TODO self._crossover_nodes, 

        self.merge_duplicated_nodes_toggle = True

        self.graphkey = None

    def mutate(self, rng=None):
        rng = np.random.default_rng(rng)
        rng.shuffle(self.mutate_methods_list)
        for mutate_method in self.mutate_methods_list:
            if mutate_method(rng=rng):
                
                if self.merge_duplicated_nodes_toggle:
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

                self.graphkey = None

        return False




    def _mutate_insert_leaf(self, rng=None):
        rng = np.random.default_rng(rng)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            for node in sorted_nodes_list:
                #if leafs are protected, check if node is a leaf
                #if node is a leaf, skip because we don't want to add node on top of node
                if (self.leaf_search_space is not None #if leafs are protected
                    and   len(list(self.graph.successors(node))) == 0 #if node is leaf
                    and  len(list(self.graph.predecessors(node))) > 0 #except if node is root, in which case we want to add a leaf even if it happens to be a leaf too
                    ):

                    continue

                #If node *is* the root or is not a leaf, add leaf node. (dont want to add leaf on top of leaf)
                if self.leaf_search_space is not None:
                    new_node = self.leaf_search_space.generate(rng)
                else:
                    new_node = self.inner_search_space.generate(rng)

                self.graph.add_node(new_node)
                self.graph.add_edge(node, new_node)
                return True

        return False
    
    def _mutate_insert_inner_node(self, rng=None):
        """
        Finds an edge in the graph and inserts a new node between the two nodes. Removes the edge between the two nodes.
        """
        rng = np.random.default_rng(rng)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            rng.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                #loop through children of node
                for child_node in list(self.graph.successors(node)):

                    if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                        if self.leaf_search_space is not None:
                            #If if we are protecting leafs, dont add connection into a leaf
                            if len(list(nx.descendants(self.graph,node))) ==0 :
                                continue

                        new_node = self.inner_search_space.generate(rng)

                        self.graph.add_node(new_node)
                        self.graph.add_edges_from([(node, new_node), (new_node, child_node)])
                        self.graph.remove_edge(node, child_node)
                        return True

        return False


    def _mutate_remove_node(self, rng=None):
        '''
        Removes a randomly chosen node and connects its parents to its children.
        If the node is the only leaf for an inner node and 'leaf_search_space' is not none, we do not remove it.
        '''
        rng = np.random.default_rng(rng)
        nodes_list = list(self.graph.nodes)
        nodes_list.remove(self.root)
        leaves = get_leaves(self.graph)

        while len(nodes_list) > 0:
            node = rng.choice(nodes_list)
            nodes_list.remove(node)

            if self.leaf_search_space is not None and len(list(nx.descendants(self.graph,node))) == 0 : #if the node is a leaf
                if len(leaves) <= 1:
                    continue #dont remove the last leaf
                leaf_parents = self.graph.predecessors(node)

                # if any of the parents of the node has one one child, continue
                if any([len(list(self.graph.successors(lp))) < 2 for lp in leaf_parents]): #dont remove a leaf if it is the only input into another node.
                    continue

                remove_and_stitch(self.graph, node)
                remove_nodes_disconnected_from_node(self.graph, self.root)
                return True

            else:
                remove_and_stitch(self.graph, node)
                remove_nodes_disconnected_from_node(self.graph, self.root)
                return True

        return False
            
        

    def _mutate_node(self, rng=None):
        '''
        Mutates the hyperparameters for a randomly chosen node in the graph.
        '''
        rng = np.random.default_rng(rng)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        completed_one = False
        for node in sorted_nodes_list:
            if node.mutate(rng):
                return True
        return False

    def _mutate_remove_edge(self, rng=None):
        '''
        Deletes an edge as long as deleting that edge does not make the graph disconnected.
        '''
        rng = np.random.default_rng(rng)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        for child_node in sorted_nodes_list:
            parents = list(self.graph.predecessors(child_node))
            if len(parents) > 1: # if it has more than one parent, you can remove an edge (if this is the only child of a node, it will become a leaf)

                for parent_node in parents:
                    # if removing the egde will make the parent_node a leaf node, skip
                    if self.leaf_search_space is not None and len(list(self.graph.successors(parent_node))) < 2:
                        continue

                    self.graph.remove_edge(parent_node, child_node)
                    return True
        return False   
    
    def _mutate_add_edge(self, rng=None):
        '''
        Randomly add an edge from a node to another node that is not an ancestor of the first node.
        '''
        rng = np.random.default_rng(rng)
        sorted_nodes_list = list(self.graph.nodes)
        rng.shuffle(sorted_nodes_list)
        for child_node in sorted_nodes_list:
            for parent_node in sorted_nodes_list:
                if self.leaf_search_space is not None:
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
        
    def _mutate_insert_bypass_node(self, rng=None):
        """
        Pick two nodes (doesn't necessarily need to be connected). Create a new node. connect one node to the new node and the new node to the other node.
        Does not remove any edges.
        """
        rng = np.random.default_rng(rng)
        if self.max_size > self.graph.number_of_nodes():
            sorted_nodes_list = list(self.graph.nodes)
            sorted_nodes_list2 = list(self.graph.nodes)
            rng.shuffle(sorted_nodes_list) #TODO: sort by number of children and/or parents? bias model one way or another
            rng.shuffle(sorted_nodes_list2)
            for node in sorted_nodes_list:
                for child_node in sorted_nodes_list2:
                    if child_node is not node and child_node not in nx.ancestors(self.graph, node):
                        if self.leaf_search_space is not None:
                            #If if we are protecting leafs, dont add connection into a leaf
                            if len(list(nx.descendants(self.graph,node))) ==0 :
                                continue

                        new_node = self.inner_search_space.generate(rng)

                        self.graph.add_node(new_node)
                        self.graph.add_edges_from([(node, new_node), (new_node, child_node)])
                        return True

        return False


    def crossover(self, ind2, rng=None):
        '''
        self is the first individual, ind2 is the second individual
        If crossover_same_depth, it will select graphindividuals at the same recursive depth.
        Otherwise, it will select graphindividuals randomly from the entire graph and its subgraphs.

        This does not impact graphs without subgraphs. And it does not impacts nodes that are not graphindividuals. Cros
        '''

        rng = np.random.default_rng(rng)

        rng.shuffle(self.crossover_methods_list)

        finished = False

        for crossover_method in self.crossover_methods_list:
            if crossover_method(ind2, rng=rng):
                self._merge_duplicated_nodes()
                finished = True
                break

        if self.__debug:
            try:
                nx.find_cycle(self.graph)
                print('something went wrong with ', crossover_method)
            except:
                pass
        
        if finished:
            self.graphkey = None

        return finished


    def _crossover_swap_branch(self, G2, rng=None):
        '''
        swaps a branch from parent1 with a branch from parent2. does not modify parent2
        '''
        rng = np.random.default_rng(rng)

        if self.crossover_same_depth:
            pair_gen = select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng=rng)
        else:
            pair_gen = select_nodes_randomly(self.graph, G2.graph, rng=rng)

        for node1, node2 in pair_gen:
            #TODO: if root is in inner_search_space, then do use it?
            if node1 is self.root or node2 is G2.root: #dont want to add root as inner node
                continue

            #check if node1 is a leaf and leafs are protected, don't add an input to the leave
            if self.leaf_search_space is not None: #if we are protecting leaves,
                node1_is_leaf = len(list(self.graph.successors(node1))) == 0
                node2_is_leaf = len(list(G2.graph.successors(node2))) == 0
                #if not ((node1_is_leaf and node1_is_leaf) or (not node1_is_leaf and not node2_is_leaf)): #if node1 is a leaf
                #if (node1_is_leaf and (not node2_is_leaf)) or ( (not node1_is_leaf) and node2_is_leaf):
                if not node1_is_leaf:
                    #only continue if node1 and node2 are both leaves or both not leaves
                    continue

            temp_graph_1 = self.graph.copy()
            temp_graph_1.remove_node(node1)
            remove_nodes_disconnected_from_node(temp_graph_1, self.root)

            #isolating the branch
            branch2 = G2.graph.copy()
            n2_descendants = nx.descendants(branch2,node2)
            for n in list(branch2.nodes):
                if n not in n2_descendants and n is not node2: #removes all nodes not in the branch
                    branch2.remove_node(n)

            branch2 = copy.deepcopy(branch2)
            branch2_root = get_roots(branch2)[0]
            temp_graph_1.add_edges_from(branch2.edges)
            for p in list(self.graph.predecessors(node1)):
                temp_graph_1.add_edge(p,branch2_root)

            if temp_graph_1.number_of_nodes() > self.max_size:
                continue

            self.graph = temp_graph_1

            return True
        return False


    def _crossover_take_branch(self, G2, rng=None):
        '''
        Takes a subgraph from Parent2 and add it to a randomly chosen node in Parent1.
        '''
        rng = np.random.default_rng(rng)

        if self.crossover_same_depth:
            pair_gen = select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng=rng)
        else:
            pair_gen = select_nodes_randomly(self.graph, G2.graph, rng=rng)

        for node1, node2 in pair_gen:
            #TODO: if root is in inner_search_space, then do use it?
            if node2 is G2.root: #dont want to add root as inner node
                continue


            #check if node1 is a leaf and leafs are protected, don't add an input to the leave
            if self.leaf_search_space is not None and len(list(self.graph.successors(node1))) == 0:
                continue

            #icheck if node2 is graph individual
            # if isinstance(node2,GraphIndividual):
            #     if not ((isinstance(node2,GraphIndividual) and ("Recursive" in self.inner_search_space or "Recursive" in self.leaf_search_space))):
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
            branch2_root = get_roots(branch2)[0]
            self.graph.add_edges_from(branch2.edges)
            self.graph.add_edge(node1,branch2_root)

            return True
        return False

    

    def _crossover_nodes(self, G2, rng=None):
        '''
        Swaps the hyperparamters of one randomly chosen node in Parent1 with the hyperparameters of randomly chosen node in Parent2.
        '''
        rng = np.random.default_rng(rng)

        if self.crossover_same_depth:
            pair_gen = select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng=rng)
        else:
            pair_gen = select_nodes_randomly(self.graph, G2.graph, rng=rng)

        for node1, node2 in pair_gen:
            
            #if both nodes are leaves
            if len(list(self.graph.successors(node1)))==0 and len(list(G2.graph.successors(node2)))==0:
                if node1.crossover(node2):
                    return True

                
            #if both nodes are inner nodes
            if len(list(self.graph.successors(node1)))>0 and len(list(G2.graph.successors(node2)))>0:
                if len(list(self.graph.predecessors(node1)))>0 and len(list(G2.graph.predecessors(node2)))>0:
                    if node1.crossover(node2):
                        return True

            #if both nodes are root nodes
            if node1 is self.root and node2 is G2.root:
                if node1.crossover(node2):
                    return True


        return False

    #not including the nodes, just their children
    #Finds leaves attached to nodes and swaps them
    def _crossover_swap_leaf_at_node(self, G2, rng=None):
        rng = np.random.default_rng(rng)

        if self.crossover_same_depth:
            pair_gen = select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng=rng)
        else:
            pair_gen = select_nodes_randomly(self.graph, G2.graph, rng=rng)

        success = False
        for node1, node2 in pair_gen:
            # if leaves are protected node1 and node2 must both be leaves or both be inner nodes
            if self.leaf_search_space is not None and not (len(list(self.graph.successors(node1)))==0 ^ len(list(G2.graph.successors(node2)))==0):
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



    #TODO edit so that G2 is not modified
    def _crossover_swap_node(self, G2, rng=None):
        '''
        Swaps randomly chosen node from Parent1 with a randomly chosen node from Parent2.
        '''
        rng = np.random.default_rng(rng)

        if self.crossover_same_depth:
            pair_gen = select_nodes_same_depth(self.graph, self.root, G2.graph, G2.root, rng=rng)
        else:
            pair_gen = select_nodes_randomly(self.graph, G2.graph, rng=rng)

        for node1, node2 in pair_gen:
            if node1 is self.root or node2 is G2.root: #TODO: allow root
                continue

            #if leaves are protected
            if self.leaf_search_space is not None:
                #if one node is a leaf, the other must be a leaf
                if not((len(list(self.graph.successors(node1)))==0) ^ (len(list(G2.graph.successors(node2)))==0)):
                    continue #only continue if both are leaves, or both are not leaves


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
    

    def _merge_duplicated_nodes(self):

        graph_changed = False
        merged = False
        while(not merged):
            node_list = list(self.graph.nodes)
            merged = True
            for node, other_node in itertools.product(node_list, node_list):
                if node is other_node:
                    continue

                #If nodes are same class/hyperparameters
                if node.unique_id() == other_node.unique_id():
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


    def export_pipeline(self):
        estimator_graph = self.graph.copy()

        #mapping = {node:node.method_class(**node.hyperparameters) for node in estimator_graph}
        label_remapping = {}
        label_to_instance = {}

        for node in estimator_graph:
            this_pipeline_node = node.export_pipeline()
            found_unique_label = False
            i=1
            while not found_unique_label:
                label = "{0}_{1}".format(this_pipeline_node.__class__.__name__, i)
                if label not in label_to_instance:
                    found_unique_label = True
                else:
                    i+=1

            label_remapping[node] = label
            label_to_instance[label] = this_pipeline_node

        estimator_graph = nx.relabel_nodes(estimator_graph, label_remapping)

        for label, instance in label_to_instance.items():
            estimator_graph.nodes[label]["instance"] = instance

        return tpot2.GraphPipeline(graph=estimator_graph, memory=self.memory, use_label_encoder=self.use_label_encoder, method=self.method, cross_val_predict_cv=self.cross_val_predict_cv)
    
    
    def plot(self):
        G = self.graph.reverse()
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


    def unique_id(self):
        if self.graphkey is None:
            #copy self.graph
            new_graph = self.graph.copy()
            for n in new_graph.nodes:
                new_graph.nodes[n]['label'] = n.unique_id()
            
            new_graph = nx.convert_node_labels_to_integers(new_graph)
            self.graphkey = GraphKey(new_graph)
        
        return self.graphkey
    

class GraphPipeline(SklearnIndividualGenerator):
    def __init__(self, 
        root_search_space: SklearnIndividualGenerator, 
        leaf_search_space: SklearnIndividualGenerator = None, 
        inner_search_space: SklearnIndividualGenerator = None, 
        max_size: int = np.inf,
        crossover_same_depth: bool = False,
        cross_val_predict_cv: Union[int, Callable] = 0, #signature function(estimator, X, y=none)
        method: str = 'auto',
        memory=None,
        use_label_encoder: bool = False):
        
        """
        Defines a search space of pipelines in the shape of a Directed Acyclic Graphs. The search spaces for root, leaf, and inner nodes can be defined separately if desired.
        Each graph will have a single root serving as the final estimator which is drawn from the `root_search_space`. If the `leaf_search_space` is defined, all leaves 
        in the pipeline will be drawn from that search space. If the `leaf_search_space` is not defined, all leaves will be drawn from the `inner_search_space`.
        Nodes that are not leaves or roots will be drawn from the `inner_search_space`. If the `inner_search_space` is not defined, there will be no inner nodes.

        `cross_val_predict_cv`, `method`, `memory`, and `use_label_encoder` are passed to the GraphPipeline object when the pipeline is exported and not directly used in the search space.

        Exports to a GraphPipeline object.

        Parameters
        ----------

        root_search_space: SklearnIndividualGenerator
            The search space for the root node of the graph. This node will be the final estimator in the pipeline.
        
        inner_search_space: SklearnIndividualGenerator, optional
            The search space for the inner nodes of the graph. If not defined, there will be no inner nodes.
        
        leaf_search_space: SklearnIndividualGenerator, optional
            The search space for the leaf nodes of the graph. If not defined, the leaf nodes will be drawn from the inner_search_space.
            
        crossover_same_depth: bool, optional
            If True, crossover will only occur between nodes at the same depth in the graph. If False, crossover will occur between nodes at any depth.
        
        cross_val_predict_cv: int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy used in inner classifiers or regressors

        method: str, optional
            The prediction method to use for the inner classifiers or regressors. If 'auto', it will try to use predict_proba, decision_function, or predict in that order.

        memory: str or object with the joblib.Memory interface, optional
            Used to cache the input and outputs of nodes to prevent refitting or computationally heavy transformations. By default, no caching is performed. If a string is given, it is the path to the caching directory.

        use_label_encoder: bool, optional
            If True, the label encoder is used to encode the labels to be 0 to N. If False, the label encoder is not used.
            Mainly useful for classifiers (XGBoost) that require labels to be ints from 0 to N.
            Can also be a sklearn.preprocessing.LabelEncoder object. If so, that label encoder is used.
            
        """


        self.root_search_space = root_search_space
        self.leaf_search_space = leaf_search_space
        self.inner_search_space = inner_search_space
        self.max_size = max_size
        self.crossover_same_depth = crossover_same_depth

        self.cross_val_predict_cv = cross_val_predict_cv
        self.method = method
        self.memory = memory
        self.use_label_encoder = use_label_encoder

    def generate(self, rng=None):
        rng = np.random.default_rng(rng)
        ind =  GraphPipelineIndividual(self.root_search_space, self.leaf_search_space, self.inner_search_space, self.max_size, self.crossover_same_depth, 
                                       self.cross_val_predict_cv, self.method, self.memory, self.use_label_encoder, rng=rng)  
            # if user specified limit, grab a random number between that limit
        
        if self.max_size is None or self.max_size == np.inf:
            n_nodes = rng.integers(1, 5)
        else:
            n_nodes = min(rng.integers(1, self.max_size), 5)
        
        starting_ops = []
        if self.inner_search_space is not None:
            starting_ops.append(ind._mutate_insert_inner_node)
        if self.leaf_search_space is not None or self.inner_search_space is not None:
            starting_ops.append(ind._mutate_insert_leaf)
            n_nodes -= 1

        if len(starting_ops) > 0:
            for _ in range(n_nodes-1):
                func = rng.choice(starting_ops)
                func(rng=rng)

        ind._merge_duplicated_nodes()

        return ind
    




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


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

from functools import partial
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import sklearn
from sklearn.utils.metaestimators import available_if
import pandas as pd

from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_memory
from sklearn.preprocessing import LabelEncoder

from sklearn.base import is_classifier, is_regressor
from sklearn.utils._tags import get_tags
import copy
#labels - str
#attributes - "instance" -> instance of the type


def plot(graph: nx.DiGraph):
    G = graph.reverse()
    try:
        pos = nx.planar_layout(G)  # positions for all nodes
    except:
        pos = nx.shell_layout(G)

    # nodes
    options = {'edgecolors': 'tab:gray', 'node_size': 800, 'alpha': 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=list(G.nodes), node_color='tab:red', **options)

    # edges
    nx.draw_networkx_edges(G, pos, width=3.0, arrows=True)

    # some math labels
    labels = {}
    for i, n in enumerate(G.nodes):
        labels[n] = n#.__class__.__name__

    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='black')

    plt.tight_layout()
    plt.axis('off')
    plt.show()


#copied from https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/ensemble/_stacking.py#L121
def _method_name(name, estimator, method):
    if estimator == 'drop':
        return None
    if method == 'auto':
        if hasattr(estimator, 'predict_proba'):
            return 'predict_proba'
        elif hasattr(estimator, 'decision_function'):
            return 'decision_function'
        else:
            return 'predict'
    else:
        if not hasattr(estimator, method):
            raise ValueError(
                'Underlying estimator {} does not implement the method {}.'.format(
                    name, method
                )
            )
        return method


def estimator_fit_transform_override_cross_val_predict(estimator, X, y, cv=5, method='auto', **fit_params):

    method = _method_name(name=estimator.__class__.__name__, estimator=estimator, method=method)
    
    if (isinstance(cv, int) and cv>1) or (not isinstance(cv, int) and cv is not None):
        preds = sklearn.model_selection.cross_val_predict(estimator=estimator, X=X, y=y, cv=cv, method=method, **fit_params)
        estimator.fit(X,y, **fit_params)
    
    
    else:
        estimator.fit(X,y, **fit_params)
        func = getattr(estimator,method)
        preds = func(X)


    return preds, estimator


# https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98/sklearn/pipeline.py#L883
def _fit_transform_one(model, X, y, fit_transform=True, **fit_params):
    """Fit and transform one step in a pipeline."""

    if fit_transform and hasattr(model, "fit_transform"):
        res = model.fit_transform(X, y, **fit_params)
    else:
        res = model.fit(X, y, **fit_params).transform(X)
        #return model


    return res, model

#TODO: make sure predict proba doesn't return p and 1-p for nclasses=2
def fit_sklearn_digraph(graph: nx.DiGraph,
        X,
        y,
        method='auto',
        cross_val_predict_cv = 0, #func(est,X,y) -> transformed_X
        memory = None,
        topo_sort = None,
        ):

    memory = check_memory(memory)


    fit_transform_one_cached = memory.cache(_fit_transform_one)
    estimator_fit_transform_override_cross_val_predict_cached = memory.cache(estimator_fit_transform_override_cross_val_predict)

    if topo_sort is None:
        topo_sort = list(nx.topological_sort(graph))
        topo_sort.reverse()

    transformed_steps =  {}

    for i in range(len(topo_sort)):
        node = topo_sort[i]
        instance = graph.nodes[node]["instance"]
        if len(list(get_ordered_successors(graph, node))) == 0: #If this node had no inputs use X
            this_X = X
        else: #in node has inputs, get those
            this_X = np.hstack([transformed_steps[child] for child in get_ordered_successors(graph, node)])

        # Removed so that the cache is the same for all models. Not including transform would index it seperately 
        #if i == len(topo_sort)-1: #last method doesn't need transformed.
        #    instance.fit(this_X, y)
        

        if is_classifier(instance) or is_regressor(instance):
            transformed, instance = estimator_fit_transform_override_cross_val_predict_cached(instance, this_X, y, cv=cross_val_predict_cv, method=method)
        else:
            transformed, instance = fit_transform_one_cached(instance, this_X, y)#instance.fit_transform(this_X,y)
        
        graph.nodes[node]["instance"] = instance

        if len(transformed.shape) == 1:
            transformed = transformed.reshape(-1, 1)

        transformed_steps[node] = transformed



#TODO add attribute to decide 'method' for each node
#TODO make more memory efficient. Free memory when a transformation is no longer needed
#TODO better handle multiple roots
def transform_sklearn_digraph(graph: nx.DiGraph,
                    X, 
                    method = 'auto',
                    output_nodes = None,
                    topo_sort = None,):

    if graph.number_of_nodes() == 1: #TODO make this better...
        return X

    if topo_sort is None:
        topo_sort = list(nx.topological_sort(graph))
        topo_sort.reverse()

    transformed_steps = {}

    for i in range(len(topo_sort)):
        node = topo_sort[i]
        instance = graph.nodes[node]["instance"]
        if len(list(get_ordered_successors(graph, node))) == 0:
            this_X = X
        else:
            this_X = np.hstack([transformed_steps[child] for child in get_ordered_successors(graph, node)])
            
        if is_classifier(instance) or is_regressor(instance):
            this_method = _method_name(instance.__class__.__name__, instance, method)
            transformed = getattr(instance, this_method)(this_X)
        else:
            transformed = instance.transform(this_X)

        if len(transformed.shape) == 1:
            transformed = transformed.reshape(-1, 1)

        transformed_steps[node] = transformed

    if output_nodes is None:
        return transformed_steps
    else:
        return {n: transformed_steps[n] for n in output_nodes}


def get_inputs_to_node(graph: nx.DiGraph,
                    X, 
                    node,
                    method = 'auto',
                    topo_sort = None,
                    ):
    
    if len(list(get_ordered_successors(graph, node))) == 0:
        this_X = X
    else:
        transformed_steps = transform_sklearn_digraph(graph,
                                        X, 
                                        method,
                                        topo_sort = topo_sort,
                                        )

        this_X = np.hstack([transformed_steps[child] for child in get_ordered_successors(graph, node)])
    return this_X
    


def _estimator_has(attr):
    '''Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted final estimator if available, otherwise we
    check the unfitted final estimator.
    '''

    def check(self):
        return hasattr(self.graph.nodes[self.root]["instance"], attr)
        
    return check

def setup_ordered_successors(graph: nx.DiGraph):
    for node in graph.nodes:
        graph.nodes[node]["successors"] = sorted(list(graph.successors(node)))

def get_ordered_successors(graph: nx.DiGraph, node):
    return graph.nodes[node]["successors"]

#TODO make sure it meets all requirements for basecomposition
class GraphPipeline(_BaseComposition):
    def __init__(
                self,
                graph,
                cross_val_predict_cv=0, #signature function(estimator, X, y=none)
                method='auto',
                memory=None,
                use_label_encoder=False,
                **kwargs,
                ):
        super().__init__(**kwargs)
        '''
        An sklearn baseestimator that uses genetic programming to optimize a pipeline.
        
        Parameters
        ----------

        graph: networkx.DiGraph
            A directed graph where the nodes are sklearn estimators and the edges are the inputs to those estimators.
        
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

        '''

        self.graph = graph
        self.cross_val_predict_cv = cross_val_predict_cv
        self.method = method
        self.memory = memory
        self.use_label_encoder = use_label_encoder

        setup_ordered_successors(graph)

        self.topo_sorted_nodes = list(nx.topological_sort(self.graph))
        self.topo_sorted_nodes.reverse()
        
        self.root = self.topo_sorted_nodes[-1]

        if self.use_label_encoder:
            if type(self.use_label_encoder) == LabelEncoder:
                self.label_encoder = self.use_label_encoder
            else:
                self.label_encoder = LabelEncoder()


        #TODO clean this up
        try:
            nx.find_cycle(self.G)
            raise BaseException 
        except: 
            pass
        
    def __str__(self):
        if len(self.graph.edges) > 0:
            return str(self.graph.edges)
        else:
            return str(self.graph.nodes)

    def fit(self, X, y):


        if self.use_label_encoder:
            if type(self.use_label_encoder) == LabelEncoder:
                y = self.label_encoder.transform(y)
            else:
                y = self.label_encoder.fit_transform(y)



        fit_sklearn_digraph(   graph=self.graph,
                                X=X,
                                y=y,
                                method=self.method,
                                cross_val_predict_cv = self.cross_val_predict_cv,
                                memory = self.memory,
                                topo_sort = self.topo_sorted_nodes,
                                )
        
        return self

    def plot(self, ):
        plot(graph = self.graph)

    def __sklearn_is_fitted__(self):
        '''Indicate whether pipeline has been fit.'''
        try:
            # check if the last step of the pipeline is fitted
            # we only check the last step since if the last step is fit, it
            # means the previous steps should also be fit. This is faster than
            # checking if every step of the pipeline is fit.
            sklearn.utils.validation.check_is_fitted(self.graph.nodes[self.root]["instance"])
            return True
        except sklearn.exceptions.NotFittedError:
            return False

    @available_if(_estimator_has('predict'))
    def predict(self, X, **predict_params):


        this_X = get_inputs_to_node(self.graph,
                    X, 
                    self.root,
                    method = self.method,
                    topo_sort = self.topo_sorted_nodes,
                    )

        preds = self.graph.nodes[self.root]["instance"].predict(this_X, **predict_params)

        if self.use_label_encoder:
            preds = self.label_encoder.inverse_transform(preds)

        return preds
    
    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X, **predict_params):


        this_X = get_inputs_to_node(self.graph,
                    X, 
                    self.root,
                    method = self.method,
                    topo_sort = self.topo_sorted_nodes,
                    )
        return self.graph.nodes[self.root]["instance"].predict_proba(this_X, **predict_params)
    
    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X, **predict_params):

        this_X = get_inputs_to_node(self.graph,
                    X, 
                    self.root,
                    method = self.method,
                    topo_sort = self.topo_sorted_nodes,
                    )
        return self.graph.nodes[self.root]["instance"].decision_function(this_X, **predict_params)
    
    @available_if(_estimator_has('transform'))
    def transform(self, X, **predict_params):
                
        this_X = get_inputs_to_node(self.graph,
                    X, 
                    self.root,
                    method = self.method,
                    topo_sort = self.topo_sorted_nodes,
                    )
        return self.graph.nodes[self.root]["instance"].transform(this_X, **predict_params)

    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        
        if self.use_label_encoder:
            return self.label_encoder.classes_
        else:
            return self.graph.nodes[self.root]["instance"].classes_

    @property
    def _estimator_type(self):
        return self.graph.nodes[self.root]["instance"]._estimator_type
    

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        final_step = self.graph.nodes[self.root]["instance"]

        try:
            last_step_tags = final_step.__sklearn_tags__()
        except:
            last_step_tags = get_tags(final_step)
        
        tags.estimator_type = last_step_tags.estimator_type
        tags.target_tags.multi_output = last_step_tags.target_tags.multi_output
        tags.classifier_tags = copy.deepcopy(last_step_tags.classifier_tags)
        tags.regressor_tags = copy.deepcopy(last_step_tags.regressor_tags)
        tags.transformer_tags = copy.deepcopy(last_step_tags.transformer_tags)

        tags.input_tags.sparse = all(
            self.graph.nodes[step]['instance'].__sklearn_tags__().input_tags.sparse
            for step in self.topo_sorted_nodes
        )

        tags.input_tags.pairwise = last_step_tags.input_tags.pairwise

        return tags
# Test all nodes have all dictionaries
import pytest
import tpot2

import tpot2
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


def test_merge_duplicate_nodes():
    knn_configspace = {}
    standard_scaler_configspace = {}

    knn_node = tpot2.search_spaces.nodes.EstimatorNode(
        method = KNeighborsClassifier,
        space = knn_configspace,
    )

    scaler_node = tpot2.search_spaces.nodes.EstimatorNode(
        method = StandardScaler,
        space = standard_scaler_configspace,
    )


    graph_search_space = tpot2.search_spaces.pipelines.GraphSearchPipeline(
    root_search_space= knn_node,
    leaf_search_space = scaler_node, 
    inner_search_space = None,
    max_size = 10,
    )

    ind = graph_search_space.generate()

    # all of these leaves should be identical
    ind._mutate_insert_leaf()
    ind._mutate_insert_leaf()
    ind._mutate_insert_leaf()
    ind._mutate_insert_leaf()

    ind._merge_duplicated_nodes()

    assert len(ind.graph.nodes) == 2
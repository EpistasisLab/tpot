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


def test_EstimatorNodeCrossover():
    knn_configspace = {}
    standard_scaler_configspace = {}

    knn_node = tpot2.search_spaces.nodes.EstimatorNode(
        method = KNeighborsClassifier,
        space = knn_configspace,
    )

    knnind1 = knn_node.generate()
    knnind2 = knn_node.generate()

    for i in range(0,10):
        knnind1.mutate()
        knnind2.mutate()
        knnind1.crossover(knnind2)


def test_ValueError_different_types():
    knn_node = tpot2.config.get_search_space(["KNeighborsClassifier"])
    sfm_wrapper_node = tpot2.config.get_search_space(["SelectFromModel_classification"])

    for i in range(10):
        ind1 = knn_node.generate()
        ind2 = sfm_wrapper_node.generate()
        assert not ind1.crossover(ind2)
        assert not ind2.crossover(ind1)

if __name__ == "__main__":
    test_EstimatorNodeCrossover()
    test_ValueError_different_types()
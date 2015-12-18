from .DecisionTree import DecisionTree
from .KNNc import KNNc
from .RandomForest import RandomForest

operator_registry = {
    'RandomForest':RandomForest(),
    'KNNc':KNNc(),
    'DecisionTree':DecisionTree()
}
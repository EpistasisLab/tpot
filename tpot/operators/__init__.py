from .DecisionTree import DecisionTree
from .KNNc import KNNc
from .LogisticRegressionGLM import LogisticRegressionGLM
from .RandomForest import RandomForest

operator_registry = {
    'DecisionTree':DecisionTree(),
    'KNNc':KNNc(),
    'LogisticRegressionGLM':LogisticRegressionGLM(),
    'RandomForest':RandomForest()
}
from .C_SVM import C_SVM
from .DecisionTree import DecisionTree
from .KNNc import KNNc
from .LogisticRegressionGLM import LogisticRegressionGLM
from .RandomForest import RandomForest

operator_registry = {
    'C_SVM':C_SVM()
    'DecisionTree':DecisionTree(),
    'KNNc':KNNc(),
    'LogisticRegressionGLM':LogisticRegressionGLM(),
    'RandomForest':RandomForest()
}
from .C_SVM import C_SVM
from .DecisionTree import DecisionTree
from .KNNc import KNNc
from .GradientBoosting import GradientBoosting
from .LogisticRegressionGLM import LogisticRegressionGLM
from .RandomForest import RandomForest

operator_registry = {
    'C_SVM':C_SVM(),
    'DecisionTree':DecisionTree(),
    'KNNc':KNNc(),
    'GradientBoosting':GradientBoosting(),
    'LogisticRegressionGLM':LogisticRegressionGLM(),
    'RandomForest':RandomForest()
}
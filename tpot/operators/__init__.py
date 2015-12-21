#### Statistical learning operators
from .C_SVM import C_SVM
from .DecisionTree import DecisionTree
from .KNNc import KNNc
from .GradientBoosting import GradientBoosting
from .LogisticRegressionGLM import LogisticRegressionGLM
from .PCA import PCAoperator
from .RandomForest import RandomForest

#### Misc transformation operators
#from .CombineDFs import CombineDFs

operator_registry = {
    #### Statistical learning operators
    'C_SVM':C_SVM(),
    'DecisionTree':DecisionTree(),
    'KNNc':KNNc(),
    'GradientBoosting':GradientBoosting(),
    'LogisticRegressionGLM':LogisticRegressionGLM(),
    'RandomForest':RandomForest(),
    
    ### Misc transformation operators
    #'CombineDFs':CombineDFs() ## Does not work properly. I think it's a problem with preprocess_arguments()
    'PCAoperator':PCAoperator()
    }
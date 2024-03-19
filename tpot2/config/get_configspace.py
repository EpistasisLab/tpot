from ..search_spaces.nodes import EstimatorNode
from ..search_spaces.pipelines import ChoicePipeline

from .classifiers import *
from .transformers import *
from .regressors import *
from .selectors import *


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


from tpot2.builtin_modules import ZeroCount, OneHotEncoder, ColumnOneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

import sklearn.feature_selection


from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression



from tpot2.builtin_modules import RFE_ExtraTreesClassifier, SelectFromModel_ExtraTreesClassifier, RFE_ExtraTreesRegressor, SelectFromModel_ExtraTreesRegressor

STRING_TO_CLASS = {
    #classifiers
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "SVC": SVC,
    "LinearSVC": LinearSVC,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "SGDClassifier": SGDClassifier,
    "MLPClassifier": MLPClassifier,
    "BernoulliNB": BernoulliNB,
    "MultinomialNB": MultinomialNB,

    #transformers
    "Binarizer": Binarizer,
    "Normalizer": Normalizer,
    "PCA": PCA,
    "ZeroCount": ZeroCount,
    "OneHotEncoder": ColumnOneHotEncoder,
    "FastICA": FastICA,
    "FeatureAgglomeration": FeatureAgglomeration,
    "Nystroem": Nystroem,
    "RBFSampler": RBFSampler,

    #selectors
    "SelectFwe": SelectFwe,
    "SelectPercentile": SelectPercentile,
    "VarianceThreshold": VarianceThreshold,
    "RFE": RFE,
    "SelectFromModel": SelectFromModel,
}




def get_configspace(name, n_classes=3, n_samples=100, random_state=None):
    match name:
        #classifiers.py
        case "LogisticRegression":
            return get_LogisticRegression_ConfigurationSpace()
        case "KNeighborsClassifier":
            return get_KNeighborsClassifier_ConfigurationSpace(n_samples=n_samples)
        case "DecisionTreeClassifier":
            return get_DecisionTreeClassifier_ConfigurationSpace()
        case "SVC":
            return get_SVC_ConfigurationSpace()
        case "LinearSVC":
            return get_LinearSVC_ConfigurationSpace()
        case "RandomForestClassifier":
            return get_RandomForestClassifier_ConfigurationSpace(random_state=random_state)
        case "GradientBoostingClassifier":
            return get_GradientBoostingClassifier_ConfigurationSpace(n_classes=n_classes)
        case "XGBClassifier":
            return get_XGBClassifier_ConfigurationSpace(random_state=random_state)
        case "LGBMClassifier":
            return get_LGBMClassifier_ConfigurationSpace(random_state=random_state)
        case "ExtraTreesClassifier":
            return get_ExtraTreesClassifier_ConfigurationSpace(random_state=random_state)
        case "SGDClassifier":
            return get_SGDClassifier_ConfigurationSpace(random_state=random_state)
        case "MLPClassifier":
            return get_MLPClassifier_ConfigurationSpace(random_state=random_state)
        case "BernoulliNB":
            return get_BernoulliNB_ConfigurationSpace()
        case "MultinomialNB":
            return get_MultinomialNB_ConfigurationSpace()
        
        #transformers.py
        case "Binarizer":
            return Binarizer_configspace
        case "Normalizer":
            return Normalizer_configspace
        case "PCA":
            return PCA_configspace
        case "ZeroCount":
            return ZeroCount_configspace
        case "OneHotEncoder":
            return OneHotEncoder_configspace
        case "FastICA":
            return get_FastICA_configspace()
        case "FeatureAgglomeration":
            return get_FeatureAgglomeration_configspace()
        case "Nystroem":
            return get_Nystroem_configspace()
        case "RBFSampler":
            return get_RBFSampler_configspace()
        
        #selectors.py
        case "SelectFwe":
            return SelectFwe_configspace 
        case "SelectPercentile":
            return SelectPercentile_configspace
        case "VarianceThreshold":
            return VarianceThreshold_configspace
        case "RFE":
            return RFE_configspace_part
        case "SelectFromModel":
            return SelectFromModel_configspace_part
   

def check_for_special(name):
    match name:
        case "selectors":
            return ["SelectFwe", "SelectPercentile", "VarianceThreshold",]
        case "classifiers":
            return ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "SVC", "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "ExtraTreesClassifier", "SGDClassifier", "MLPClassifier", "BernoulliNB", "MultinomialNB"]
        case "transformers":
            return ["Binarizer", "Normalizer", "PCA", "ZeroCount", "OneHotEncoder", "FastICA", "FeatureAgglomeration", "Nystroem", "RBFSampler"]
    
    return name


def get_search_space(name, n_classes=3, n_samples=100, random_state=None):
    name = check_for_special(name)

    #if list of names, return a list of EstimatorNodes
    if isinstance(name, list) or isinstance(name, np.ndarray):
        search_spaces = [get_search_space(n, n_classes=n_classes, n_samples=n_samples, random_state=random_state) for n in name]
        return ChoicePipeline(choice_list=search_spaces)
    else:
        return get_estimatornode(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)


def get_estimatornode(name, n_classes=3, n_samples=100, random_state=None):
    configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
    
    
    return EstimatorNode(STRING_TO_CLASS[name], configspace)

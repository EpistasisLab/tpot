from ..search_spaces.nodes import EstimatorNode
from ..search_spaces.pipelines import ChoicePipeline, WrapperPipeline

from . import classifiers
from . import transformers
from . import selectors
from . import regressors
from . import autoqtl_builtins
from . import imputers
from . import mdr_configs
from . import special_configs

import numpy as np

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


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import RidgeCV


from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV

from xgboost import XGBRegressor


from tpot2.builtin_modules import RFE_ExtraTreesClassifier, SelectFromModel_ExtraTreesClassifier, RFE_ExtraTreesRegressor, SelectFromModel_ExtraTreesRegressor


all_methods = [SGDClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, MLPClassifier, DecisionTreeClassifier, XGBClassifier, KNeighborsClassifier, SVC, LogisticRegression, LGBMClassifier, LinearSVC, GaussianNB, BernoulliNB, MultinomialNB, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, DecisionTreeRegressor, KNeighborsRegressor, XGBRegressor, RFE_ExtraTreesClassifier, SelectFromModel_ExtraTreesClassifier, RFE_ExtraTreesRegressor, SelectFromModel_ExtraTreesRegressor, ZeroCount, OneHotEncoder, ColumnOneHotEncoder, Binarizer, FastICA, FeatureAgglomeration, MaxAbsScaler, MinMaxScaler, Normalizer, Nystroem, PCA, PolynomialFeatures, RBFSampler, RobustScaler, StandardScaler, SelectFwe, SelectPercentile, VarianceThreshold, RFE, SelectFromModel, f_classif, f_regression, SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, SVR, LinearSVR, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, DecisionTreeRegressor, KNeighborsRegressor, ElasticNetCV,
               ]

STRING_TO_CLASS = {
    t.__name__: t for t in all_methods
}


GROUPNAMES = {
        "selectors": ["SelectFwe", "SelectPercentile", "VarianceThreshold",],
        "selectors_classification": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_classification", "SelectFromModel_classification"],
        "selectors_regression": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_regression", "SelectFromModel_regression"],
        "classifiers" :  ["BernoulliNB", "DecisionTreeClassifier", "ExtraTreesClassifier", "GaussianNB", "GradientBoostingClassifier", "KNeighborsClassifier", "LinearDiscriminantAnalysis", "LinearSVC", "QuadraticDiscriminantAnalysis", "PassiveAggressiveClassifier", "LogisticRegression", "MLPClassifier", "MultinomialNB", "PassiveAggressiveClassifier", "Perceptron", "QuadraticDiscriminantAnalysis", "RandomForestClassifier", "RidgeClassifier", "SGDClassifier", "SVC", "XGBClassifier", "LGBMClassifier"],
        "transformers":  ["Binarizer", "Normalizer", "PCA", "ZeroCount", "OneHotEncoder", "FastICA", "FeatureAgglomeration", "Nystroem", "RBFSampler"],
}



def get_configspace(name, n_classes=3, n_samples=100, random_state=None):
    match name:
        #classifiers.py
        case "LogisticRegression":
            return classifiers.get_LogisticRegression_ConfigurationSpace()
        case "KNeighborsClassifier":
            return classifiers.get_KNeighborsClassifier_ConfigurationSpace(n_samples=n_samples)
        case "DecisionTreeClassifier":
            return classifiers.get_DecisionTreeClassifier_ConfigurationSpace()
        case "SVC":
            return classifiers.get_SVC_ConfigurationSpace()
        case "LinearSVC":
            return classifiers.get_LinearSVC_ConfigurationSpace()
        case "RandomForestClassifier":
            return classifiers.get_RandomForestClassifier_ConfigurationSpace(random_state=random_state)
        case "GradientBoostingClassifier":
            return classifiers.get_GradientBoostingClassifier_ConfigurationSpace(n_classes=n_classes)
        case "XGBClassifier":
            return classifiers.get_XGBClassifier_ConfigurationSpace(random_state=random_state)
        case "LGBMClassifier":
            return classifiers.get_LGBMClassifier_ConfigurationSpace(random_state=random_state)
        case "ExtraTreesClassifier":
            return classifiers.get_ExtraTreesClassifier_ConfigurationSpace(random_state=random_state)
        case "SGDClassifier":
            return classifiers.get_SGDClassifier_ConfigurationSpace(random_state=random_state)
        case "MLPClassifier":
            return classifiers.get_MLPClassifier_ConfigurationSpace(random_state=random_state)
        case "BernoulliNB":
            return classifiers.get_BernoulliNB_ConfigurationSpace()
        case "MultinomialNB":
            return classifiers.get_MultinomialNB_ConfigurationSpace()
        
        #transformers.py
        case "Binarizer":
            return transformers.Binarizer_configspace
        case "Normalizer":
            return transformers.Normalizer_configspace
        case "PCA":
            return transformers.PCA_configspace
        case "ZeroCount":
            return transformers.ZeroCount_configspace
        case "OneHotEncoder":
            return transformers.OneHotEncoder_configspace
        case "FastICA":
            return transformers.get_FastICA_configspace()
        case "FeatureAgglomeration":
            return transformers.get_FeatureAgglomeration_configspace()
        case "Nystroem":
            return transformers.get_Nystroem_configspace()
        case "RBFSampler":
            return transformers.get_RBFSampler_configspace()
        
        #selectors.py
        case "SelectFwe":
            return selectors.SelectFwe_configspace 
        case "SelectPercentile":
            return selectors.SelectPercentile_configspace
        case "VarianceThreshold":
            return selectors.VarianceThreshold_configspace
        case "RFE":
            return selectors.RFE_configspace_part
        case "SelectFromModel":
            return selectors.SelectFromModel_configspace_part
        
    return None
   

def get_search_space(name, n_classes=3, n_samples=100, random_state=None):
    name = GROUPNAMES[name]

    if name is None:
        return None

    if name not in STRING_TO_CLASS:
        return None

    #if list of names, return a list of EstimatorNodes
    if isinstance(name, list) or isinstance(name, np.ndarray):
        search_spaces = [get_search_space(n, n_classes=n_classes, n_samples=n_samples, random_state=random_state) for n in name]
        #remove Nones
        search_spaces = [s for s in search_spaces if s is not None]
        
        return ChoicePipeline(choice_list=search_spaces)
    else:
        return get_node(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)


def get_node(name, n_classes=3, n_samples=100, random_state=None):

    #these are wrappers
    if name == "RFE_classification":
        rfe_sp = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesClassifier", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(nodegen=ext, method=RFE, configspace=rfe_sp)
    if name == "RFE_regression":
        rfe_sp = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesRegressor", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(nodegen=ext, method=RFE, configspace=rfe_sp)
    if name == "SelectFromModel_classification":
        sfm_sp = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesClassifier", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(nodegen=ext, method=SelectFromModel, configspace=sfm_sp)
    if name == "SelectFromModel_regression":
        sfm_sp = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesRegressor", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(nodegen=ext, method=SelectFromModel, configspace=sfm_sp)


    configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
    return EstimatorNode(STRING_TO_CLASS[name], configspace)

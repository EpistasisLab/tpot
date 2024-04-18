import importlib.util
import sys
import numpy as np
import warnings

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

from . import classifiers_sklearnex
from . import regressors_sklearnex

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

#autoqtl_builtins
from tpot2.builtin_modules import genetic_encoders
from tpot2.builtin_modules import feature_encoding_frequency_selector

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
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

import sklearn.feature_selection

#TODO create a selectomixin using these?
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars, LassoLarsCV
from sklearn.linear_model import RidgeCV

from sklearn.svm import SVR, SVC
from sklearn.svm import LinearSVR, LinearSVC

from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor,RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.gaussian_process import GaussianProcessRegressor

from xgboost import XGBRegressor


from tpot2.builtin_modules import AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer


#MDR


all_methods = [SGDClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, MLPClassifier, DecisionTreeClassifier, XGBClassifier, KNeighborsClassifier, SVC, LogisticRegression, LGBMClassifier, LinearSVC, GaussianNB, BernoulliNB, MultinomialNB, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, DecisionTreeRegressor, KNeighborsRegressor, XGBRegressor,  ZeroCount, OneHotEncoder, ColumnOneHotEncoder, Binarizer, FastICA, FeatureAgglomeration, MaxAbsScaler, MinMaxScaler, Normalizer, Nystroem, PCA, PolynomialFeatures, RBFSampler, RobustScaler, StandardScaler, SelectFwe, SelectPercentile, VarianceThreshold, SGDRegressor, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, SVR, LinearSVR, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, DecisionTreeRegressor, KNeighborsRegressor, ElasticNetCV,
               AdaBoostClassifier,
               GaussianProcessRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
               AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer,
               PowerTransformer, QuantileTransformer,
               ]


#if mdr is installed
if 'mdr' in sys.modules:
    from mdr import MDR, ContinuousMDR
    all_methods.append(MDR)
    all_methods.append(ContinuousMDR)

if 'skrebate' in sys.modules:
    from skrebate import ReliefF, SURF, SURFstar, MultiSURF
    all_methods.append(ReliefF)
    all_methods.append(SURF)
    all_methods.append(SURFstar)
    all_methods.append(MultiSURF)

if 'sklearnex' in sys.modules:
    import sklearnex

    all_methods.append(sklearnex.linear_model.LinearRegression)
    all_methods.append(sklearnex.linear_model.Ridge)
    all_methods.append(sklearnex.linear_model.Lasso)
    all_methods.append(sklearnex.linear_model.ElasticNet)
    all_methods.append(sklearnex.svm.SVR)
    all_methods.append(sklearnex.svm.NuSVR)
    all_methods.append(sklearnex.ensemble.RandomForestRegressor)
    all_methods.append(sklearnex.neighbors.KNeighborsRegressor)
    all_methods.append(sklearnex.ensemble.RandomForestClassifier)
    all_methods.append(sklearnex.neighbors.KNeighborsClassifier)
    all_methods.append(sklearnex.svm.SVC)
    all_methods.append(sklearnex.svm.NuSVC)
    all_methods.append(sklearnex.linear_model.LogisticRegression)


STRING_TO_CLASS = {
    t.__name__: t for t in all_methods
}


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor

GROUPNAMES = {
        "selectors": ["SelectFwe", "SelectPercentile", "VarianceThreshold",],
        "selectors_classification": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_classification", "SelectFromModel_classification"],
        "selectors_regression": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_regression", "SelectFromModel_regression"],
        "classifiers" :  ['AdaBoostClassifier', 'BernoulliNB', 'DecisionTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'HistGradientBoostingClassifier', 'KNeighborsClassifier', 'LogisticRegression', "LinearSVC", "SVC", 'MLPClassifier', 'MultinomialNB', "PassiveAggressiveClassifier", "QuadraticDiscriminantAnalysis", 'RandomForestClassifier', 'SGDClassifier', 'XGBClassifier'],
        "regressors" : ['AdaBoostRegressor', "ARDRegression", 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'GaussianProcessRegressor', 'HistGradientBoostingRegressor', 'KNeighborsRegressor', 'LinearDiscriminantAnalysis', 'LinearSVR', "MLPRegressor", 'RandomForestRegressor', 'SGDRegressor', 'SVR', 'XGBRegressor'],
        "transformers":  ["Binarizer", "Normalizer", "PCA", "ZeroCount", "OneHotEncoder", "FastICA", "FeatureAgglomeration", "Nystroem", "RBFSampler", "QuantileTransformer", "PowerTransformer"],
        "arithmatic": ["AddTransformer", "mul_neg_1_Transformer", "MulTransformer", "SafeReciprocalTransformer", "EQTransformer", "NETransformer", "GETransformer", "GTTransformer", "LETransformer", "LTTransformer", "MinTransformer", "MaxTransformer"],
        "imputers": [],
        "skrebate": ["ReliefF", "SURF", "SURFstar", "MultiSURF"],
        "genetic_encoders": ["DominantEncoder", "RecessiveEncoder", "HeterosisEncoder", "UnderDominanceEncoder", "OverDominanceEncoder"],

        "classifiers_sklearnex" : ["RandomForestClassifier_sklearnex", "LogisticRegression_sklearnex", "KNeighborsClassifier_sklearnex", "SVC_sklearnex","NuSVC_sklearnex"],
        "regressors_sklearnex" : ["LinearRegression_sklearnex", "Ridge_sklearnex", "Lasso_sklearnex", "ElasticNet_sklearnex", "SVR_sklearnex", "NuSVR_sklearnex", "RandomForestRegressor_sklearnex", "KNeighborsRegressor_sklearnex"],
}



def get_configspace(name, n_classes=3, n_samples=100, n_features=100, random_state=None):
    match name:

        #autoqtl_builtins.py
        case "FeatureEncodingFrequencySelector":
            return autoqtl_builtins.FeatureEncodingFrequencySelector_ConfigurationSpace
        case "DominantEncoder":
            return {}
        case "RecessiveEncoder":
            return {}
        case "HeterosisEncoder":
            return {}
        case "UnderDominanceEncoder":
            return {}
        case "OverDominanceEncoder":
            return {}


        #classifiers.py
        case "AdaBoostClassifier":
            return classifiers.get_AdaBoostClassifier_ConfigurationSpace(random_state=random_state)
        case "LogisticRegression":
            return classifiers.get_LogisticRegression_ConfigurationSpace(n_samples=n_samples, n_features=n_features, random_state=random_state)
        case "KNeighborsClassifier":
            return classifiers.get_KNeighborsClassifier_ConfigurationSpace(n_samples=n_samples)
        case "DecisionTreeClassifier":
            return classifiers.get_DecisionTreeClassifier_ConfigurationSpace(n_featues=n_features, random_state=random_state)
        case "SVC":
            return classifiers.get_SVC_ConfigurationSpace(random_state=random_state)
        case "LinearSVC":
            return classifiers.get_LinearSVC_ConfigurationSpace(random_state=random_state)
        case "RandomForestClassifier":
            return classifiers.get_RandomForestClassifier_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "GradientBoostingClassifier":
            return classifiers.get_GradientBoostingClassifier_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "HistGradientBoostingClassifier":
            return classifiers.get_HistGradientBoostingClassifier_ConfigurationSpace(n_features=n_features, random_state=random_state)
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
        case "GaussianNB":
            return {}
        case "LassoLarsCV":
            return {}
        case "ElasticNetCV":
            return regressors.ElasticNetCV_configspace
        case "RidgeCV":
            return {}

        #regressors.py
        case "RandomForestRegressor":
            return regressors.get_RandomForestRegressor_ConfigurationSpace(random_state=random_state)
        case "SGDRegressor":
            return regressors.get_SGDRegressor_ConfigurationSpace(random_state=random_state)
        case "Ridge":
            return regressors.get_Ridge_ConfigurationSpace(random_state=random_state)
        case "Lasso":
            return regressors.get_Lasso_ConfigurationSpace(random_state=random_state)
        case "ElasticNet":
            return regressors.get_ElasticNet_ConfigurationSpace(random_state=random_state)
        case "Lars":
            return regressors.get_Lars_ConfigurationSpace(random_state=random_state)
        case "OthogonalMatchingPursuit":
            return regressors.get_OthogonalMatchingPursuit_ConfigurationSpace()
        case "BayesianRidge":
            return regressors.get_BayesianRidge_ConfigurationSpace()
        case "LassoLars":
            return regressors.get_LassoLars_ConfigurationSpace(random_state=random_state)
        case "BaggingRegressor":
            return regressors.get_BaggingRegressor_ConfigurationSpace(random_state=random_state)
        case "ARDRegression":
            return regressors.get_ARDRegression_ConfigurationSpace()
        case "TheilSenRegressor":
            return regressors.get_TheilSenRegressor_ConfigurationSpace(random_state=random_state)
        case "Perceptron":
            return regressors.get_Perceptron_ConfigurationSpace(random_state=random_state)
        case "DecisionTreeRegressor":
            return regressors.get_DecisionTreeRegressor_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "LinearSVR":
            return regressors.get_LinearSVR_ConfigurationSpace(random_state=random_state)
        case "SVR":
            return regressors.get_SVR_ConfigurationSpace()
        case "XGBRegressor":
            return regressors.get_XGBRegressor_ConfigurationSpace(random_state=random_state)
        case "AdaBoostRegressor":
            return regressors.get_AdaBoostRegressor_ConfigurationSpace(random_state=random_state)
        case "ExtraTreesRegressor":
            return regressors.get_ExtraTreesRegressor_ConfigurationSpace(random_state=random_state)
        case "GradientBoostingRegressor":
            return regressors.get_GradientBoostingRegressor_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "HistGradientBoostingRegressor":
            return regressors.get_HistGradientBoostingRegressor_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "MLPRegressor":
            return regressors.get_MLPRegressor_ConfigurationSpace(random_state=random_state)
        case "KNeighborsRegressor":
            return regressors.get_KNeighborsRegressor_ConfigurationSpace(n_samples=n_samples)
        case "GaussianProcessRegressor":
            return regressors.get_GaussianProcessRegressor_ConfigurationSpace(n_features=n_features, random_state=random_state)

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
            return transformers.get_FastICA_configspace(n_features=n_features, random_state=random_state)
        case "FeatureAgglomeration":
            return transformers.get_FeatureAgglomeration_configspace(n_features=n_features,)
        case "Nystroem":
            return transformers.get_Nystroem_configspace(n_features=n_features, random_state=random_state)
        case "RBFSampler":
            return transformers.get_RBFSampler_configspace(n_features=n_features, random_state=random_state)
        case "MinMaxScaler":
            return {}
        case "PowerTransformer":
            return {}
        case "QuantileTransformer":
            return transformers.get_QuantileTransformer_configspace(random_state=random_state)
        case "RobustScaler":
            return transformers.RobustScaler_configspace
        case "ColumnOneHotEncoder":
            return {}
        case "MaxAbsScaler":
            return {}
        case "PolynomialFeatures":
            return transformers.PolynomialFeatures_configspace
        case "StandardScaler":
            return {}

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

        
        #special_configs.py
        case "AddTransformer":
            return {}
        case "mul_neg_1_Transformer":
            return {}
        case "MulTransformer":
            return {}
        case "SafeReciprocalTransformer":
            return {}
        case "EQTransformer":
            return {}
        case "NETransformer":
            return {}
        case "GETransformer":
            return {}
        case "GTTransformer":
            return {}
        case "LETransformer":
            return {}
        case "LTTransformer":
            return {}
        case "MinTransformer":
            return {}
        case "MaxTransformer":
            return {}
        case "ZeroTransformer":
            return {}
        case "OneTransformer":
            return {}
        case "NTransformer":
            return ConfigurationSpace(

                space = {

                    'n': Float("n", bounds=(-1e3, 1e3)),
                }
            ) 
        
        #imputers.py

        #mdr_configs.py
        case "MDR":
            return mdr_configs.MDR_configspace
        case "ContinuousMDR":
            return mdr_configs.MDR_configspace
        case "ReliefF":
            return mdr_configs.get_skrebate_ReliefF_config_space(n_features=n_features)
        case "SURF":
            return mdr_configs.get_skrebate_SURF_config_space(n_features=n_features)
        case "SURFstar":
            return mdr_configs.get_skrebate_SURFstar_config_space(n_features=n_features)
        case "MultiSURF":
            return mdr_configs.get_skrebate_MultiSURF_config_space(n_features=n_features)

        #classifiers_sklearnex.py
        case "RandomForestClassifier_sklearnex":
            return classifiers_sklearnex.get_RandomForestClassifier_ConfigurationSpace(random_state=random_state)
        case "LogisticRegression_sklearnex":
            return classifiers_sklearnex.get_LogisticRegression_ConfigurationSpace(random_state=random_state)
        case "KNeighborsClassifier_sklearnex":
            return classifiers_sklearnex.get_KNeighborsClassifier_ConfigurationSpace(n_samples=n_samples)
        case "SVC_sklearnex":
            return classifiers_sklearnex.get_SVC_ConfigurationSpace(random_state=random_state)
        case "NuSVC_sklearnex":
            return classifiers_sklearnex.get_NuSVC_ConfigurationSpace(random_state=random_state)
        
        #regressors_sklearnex.py
        case "LinearRegression_sklearnex":
            return {}
        case "Ridge_sklearnex":
            return regressors_sklearnex.get_Ridge_ConfigurationSpace(random_state=random_state)
        case "Lasso_sklearnex":
            return regressors_sklearnex.get_Lasso_ConfigurationSpace(random_state=random_state)
        case "ElasticNet_sklearnex":
            return regressors_sklearnex.get_ElasticNet_ConfigurationSpace(random_state=random_state)
        case "SVR_sklearnex":
            return regressors_sklearnex.get_SVR_ConfigurationSpace(random_state=random_state)
        case "NuSVR_sklearnex":
            return regressors_sklearnex.get_NuSVR_ConfigurationSpace(random_state=random_state)
        case "RandomForestRegressor_sklearnex":
            return regressors_sklearnex.get_RandomForestRegressor_ConfigurationSpace(random_state=random_state)
        case "KNeighborsRegressor_sklearnex":
            return regressors_sklearnex.get_KNeighborsRegressor_ConfigurationSpace(n_samples=n_samples)

    #raise error
    raise ValueError(f"Could not find configspace for {name}")
   

def get_search_space(name, n_classes=3, n_samples=100, n_features=100, random_state=None):


    #if list of names, return a list of EstimatorNodes
    if isinstance(name, list) or isinstance(name, np.ndarray):
        search_spaces = [get_search_space(n, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state) for n in name]
        #remove Nones
        search_spaces = [s for s in search_spaces if s is not None]
        return ChoicePipeline(choice_list=search_spaces)
    
    if name in GROUPNAMES:
        name_list = GROUPNAMES[name]
        return get_search_space(name_list, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)

    if name is None:
        warnings.warn(f"name is None")
        return None

    if name not in STRING_TO_CLASS:
        print("FOOO ", name)
        warnings.warn(f"Could not find class for {name}")
        return None
    
    return get_node(name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)


def get_node(name, n_classes=3, n_samples=100, n_features=100, random_state=None):

    #these are wrappers that take in another estimator as a parameter
    # TODO Add AdaBoostRegressor, AdaBoostClassifier as wrappers? wrap a decision tree with different params?
    # TODO add other meta-estimators?
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
    
    #these are nodes that have special search spaces which require custom parsing of the hyperparameters
    if name == "RobustScaler":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=transformers.robust_scaler_hyperparameter_parser)
    if name == "GradientBoostingClassifier" or name == "HistGradientBoosting":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.GradientBoostingClassifier_hyperparameter_parser)
    if name == "GradientBoostingRegressor" or name == "HistGradientBoostingRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.GradientBoostingRegressor_hyperparameter_parser)
    if name == "MLPClassifier":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.MLPClassifier_hyperparameter_parser)
    if name == "MLPRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.MLPRegressor_hyperparameter_parser)
    if name == "GaussianProcessRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.GaussianProcessRegressor_hyperparameter_parser)

    configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)
    if configspace is None:
        #raise warning
        warnings.warn(f"Could not find configspace for {name}")
        return None
    
    return EstimatorNode(STRING_TO_CLASS[name], configspace)

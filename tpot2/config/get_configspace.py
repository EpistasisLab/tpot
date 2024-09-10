import importlib.util
import sys
import numpy as np
import warnings
import importlib.util

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
from tpot2.builtin_modules import genetic_encoders, feature_encoding_frequency_selector
from tpot2.builtin_modules import AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer
from tpot2.builtin_modules.genetic_encoders import DominantEncoder, RecessiveEncoder, HeterosisEncoder, UnderDominanceEncoder, OverDominanceEncoder 
from tpot2.builtin_modules import ZeroCount, ColumnOneHotEncoder, PassKBinsDiscretizer
from tpot2.builtin_modules import Passthrough, SkipTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, ElasticNetCV, PassiveAggressiveClassifier, ARDRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor,RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, PolynomialFeatures, Normalizer, MinMaxScaler, MaxAbsScaler, Binarizer
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold, RFE, SelectFromModel
from sklearn.feature_selection import f_classif, f_regression #TODO create a selectomixin using these?
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

all_methods = [SGDClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, MLPClassifier, DecisionTreeClassifier, XGBClassifier, KNeighborsClassifier, SVC, LogisticRegression, LGBMClassifier, LinearSVC, GaussianNB, BernoulliNB, MultinomialNB, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, DecisionTreeRegressor, KNeighborsRegressor, XGBRegressor,  ZeroCount, ColumnOneHotEncoder, Binarizer, FastICA, FeatureAgglomeration, MaxAbsScaler, MinMaxScaler, Normalizer, Nystroem, PCA, PolynomialFeatures, RBFSampler, RobustScaler, StandardScaler, SelectFwe, SelectPercentile, VarianceThreshold, SGDRegressor, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, SVR, LinearSVR, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, DecisionTreeRegressor, KNeighborsRegressor, ElasticNetCV,
               AdaBoostClassifier,MLPRegressor,
               GaussianProcessRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
               AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer,
               PowerTransformer, QuantileTransformer,ARDRegression, QuadraticDiscriminantAnalysis, PassiveAggressiveClassifier, LinearDiscriminantAnalysis,
               DominantEncoder, RecessiveEncoder, HeterosisEncoder, UnderDominanceEncoder, OverDominanceEncoder,
               GaussianProcessClassifier, BaggingClassifier,LGBMRegressor,
               Passthrough,SkipTransformer,
               PassKBinsDiscretizer,
               SimpleImputer, IterativeImputer, KNNImputer
               ]


#if mdr is installed
if importlib.util.find_spec('mdr') is not None:
    from mdr import MDR, ContinuousMDR
    all_methods.append(MDR)
    all_methods.append(ContinuousMDR)

if importlib.util.find_spec('skrebate') is not None:
    from skrebate import ReliefF, SURF, SURFstar, MultiSURF
    all_methods.append(ReliefF)
    all_methods.append(SURF)
    all_methods.append(SURFstar)
    all_methods.append(MultiSURF)

STRING_TO_CLASS = {
    t.__name__: t for t in all_methods
}

if importlib.util.find_spec('sklearnex') is not None:
    import sklearnex
    import sklearnex.linear_model
    import sklearnex.svm
    import sklearnex.ensemble
    import sklearnex.neighbors


    sklearnex_methods = []

    sklearnex_methods.append(sklearnex.linear_model.LinearRegression)
    sklearnex_methods.append(sklearnex.linear_model.Ridge)
    sklearnex_methods.append(sklearnex.linear_model.Lasso)
    sklearnex_methods.append(sklearnex.linear_model.ElasticNet)
    sklearnex_methods.append(sklearnex.svm.SVR)
    sklearnex_methods.append(sklearnex.svm.NuSVR)
    sklearnex_methods.append(sklearnex.ensemble.RandomForestRegressor)
    sklearnex_methods.append(sklearnex.neighbors.KNeighborsRegressor)
    sklearnex_methods.append(sklearnex.ensemble.RandomForestClassifier)
    sklearnex_methods.append(sklearnex.neighbors.KNeighborsClassifier)
    sklearnex_methods.append(sklearnex.svm.SVC)
    sklearnex_methods.append(sklearnex.svm.NuSVC)
    sklearnex_methods.append(sklearnex.linear_model.LogisticRegression)

    STRING_TO_CLASS.update({f"{t.__name__}_sklearnex": t for t in sklearnex_methods})





# not including "PassiveAggressiveClassifier" in classifiers since it is mainly for larger than memory datasets/online use cases

# TODO need to subclass "GaussianProcessClassifier" and 'GaussianProcessRegressor'. These require n_features as a parameter for the kernel, but n_features may be different depending on selection functions or transformations previously in the pipeline.

GROUPNAMES = {
        "selectors": ["SelectFwe", "SelectPercentile", "VarianceThreshold",],
        "selectors_classification": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_classification", "SelectFromModel_classification"],
        "selectors_regression": ["SelectFwe", "SelectPercentile", "VarianceThreshold", "RFE_regression", "SelectFromModel_regression"],
        "classifiers" :  ["LGBMClassifier", "BaggingClassifier", 'AdaBoostClassifier', 'BernoulliNB', 'DecisionTreeClassifier', 'ExtraTreesClassifier', 'GaussianNB', 'HistGradientBoostingClassifier', 'KNeighborsClassifier','LinearDiscriminantAnalysis', 'LogisticRegression', "LinearSVC", "SVC", 'MLPClassifier', 'MultinomialNB',  "QuadraticDiscriminantAnalysis", 'RandomForestClassifier', 'SGDClassifier', 'XGBClassifier'],
        "regressors" : ["LGBMRegressor", 'AdaBoostRegressor', "ARDRegression", 'DecisionTreeRegressor', 'ExtraTreesRegressor', 'HistGradientBoostingRegressor', 'KNeighborsRegressor',  'LinearSVR', "MLPRegressor", 'RandomForestRegressor', 'SGDRegressor', 'SVR', 'XGBRegressor'],
        
        
        "transformers":  ["PassKBinsDiscretizer", "Binarizer", "PCA", "ZeroCount", "ColumnOneHotEncoder", "FastICA", "FeatureAgglomeration", "Nystroem", "RBFSampler", "QuantileTransformer", "PowerTransformer"],
        "scalers": ["MinMaxScaler", "RobustScaler", "StandardScaler", "MaxAbsScaler", "Normalizer", ],
        "all_transformers" : ["transformers", "scalers"],

        "arithmatic": ["AddTransformer", "mul_neg_1_Transformer", "MulTransformer", "SafeReciprocalTransformer", "EQTransformer", "NETransformer", "GETransformer", "GTTransformer", "LETransformer", "LTTransformer", "MinTransformer", "MaxTransformer"],
        "imputers": ["SimpleImputer", "IterativeImputer", "KNNImputer"],
        "skrebate": ["ReliefF", "SURF", "SURFstar", "MultiSURF"],
        "genetic_encoders": ["DominantEncoder", "RecessiveEncoder", "HeterosisEncoder", "UnderDominanceEncoder", "OverDominanceEncoder"],

        "classifiers_sklearnex" : ["RandomForestClassifier_sklearnex", "LogisticRegression_sklearnex", "KNeighborsClassifier_sklearnex", "SVC_sklearnex","NuSVC_sklearnex"],
        "regressors_sklearnex" : ["LinearRegression_sklearnex", "Ridge_sklearnex", "Lasso_sklearnex", "ElasticNet_sklearnex", "SVR_sklearnex", "NuSVR_sklearnex", "RandomForestRegressor_sklearnex", "KNeighborsRegressor_sklearnex"],
}



def get_configspace(name, n_classes=3, n_samples=1000, n_features=100, random_state=None):
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

        case "Passthrough":
            return {}
        case "SkipTransformer":
            return {}

        #classifiers.py
        case "LinearDiscriminantAnalysis":
            return classifiers.get_LinearDiscriminantAnalysis_ConfigurationSpace()
        case "AdaBoostClassifier":
            return classifiers.get_AdaBoostClassifier_ConfigurationSpace(random_state=random_state)
        case "LogisticRegression":
            return classifiers.get_LogisticRegression_ConfigurationSpace(random_state=random_state)
        case "KNeighborsClassifier":
            return classifiers.get_KNeighborsClassifier_ConfigurationSpace(n_samples=n_samples)
        case "DecisionTreeClassifier":
            return classifiers.get_DecisionTreeClassifier_ConfigurationSpace(n_featues=n_features, random_state=random_state)
        case "SVC":
            return classifiers.get_SVC_ConfigurationSpace(random_state=random_state)
        case "LinearSVC":
            return classifiers.get_LinearSVC_ConfigurationSpace(random_state=random_state)
        case "RandomForestClassifier":
            return classifiers.get_RandomForestClassifier_ConfigurationSpace(random_state=random_state)
        case "GradientBoostingClassifier":
            return classifiers.get_GradientBoostingClassifier_ConfigurationSpace(n_classes=n_classes, random_state=random_state)
        case "HistGradientBoostingClassifier":
            return classifiers.get_HistGradientBoostingClassifier_ConfigurationSpace(random_state=random_state)
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
        case "PassiveAggressiveClassifier":
            return classifiers.get_PassiveAggressiveClassifier_ConfigurationSpace(random_state=random_state)
        case "QuadraticDiscriminantAnalysis":
            return classifiers.get_QuadraticDiscriminantAnalysis_ConfigurationSpace()
        case "GaussianProcessClassifier":
            return classifiers.get_GaussianProcessClassifier_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "BaggingClassifier":
            return classifiers.get_BaggingClassifier_ConfigurationSpace(random_state=random_state)

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
            return regressors.get_DecisionTreeRegressor_ConfigurationSpace(random_state=random_state)
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
            return regressors.get_GradientBoostingRegressor_ConfigurationSpace(random_state=random_state)
        case "HistGradientBoostingRegressor":
            return regressors.get_HistGradientBoostingRegressor_ConfigurationSpace(random_state=random_state)
        case "MLPRegressor":
            return regressors.get_MLPRegressor_ConfigurationSpace(random_state=random_state)
        case "KNeighborsRegressor":
            return regressors.get_KNeighborsRegressor_ConfigurationSpace(n_samples=n_samples)
        case "GaussianProcessRegressor":
            return regressors.get_GaussianProcessRegressor_ConfigurationSpace(n_features=n_features, random_state=random_state)
        case "LGBMRegressor":
            return regressors.get_LGBMRegressor_ConfigurationSpace(random_state=random_state)
        case "BaggingRegressor":
            return regressors.get_BaggingRegressor_ConfigurationSpace(random_state=random_state)

        #transformers.py
        case "Binarizer":
            return transformers.Binarizer_configspace
        case "Normalizer":
            return transformers.Normalizer_configspace
        case "PCA":
            return transformers.PCA_configspace
        case "ZeroCount":
            return transformers.ZeroCount_configspace
        case "FastICA":
            return transformers.get_FastICA_configspace(n_features=n_features, random_state=random_state)
        case "FeatureAgglomeration":
            return transformers.get_FeatureAgglomeration_configspace(n_samples=n_samples)
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
        case "PassKBinsDiscretizer":
            return transformers.get_passkbinsdiscretizer_configspace(random_state=random_state)

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

                    'n': Float("n", bounds=(-1e2, 1e2)),
                }
            ) 
        
        #imputers.py
        case "SimpleImputer":
            return imputers.simple_imputer_cs
        case "IterativeImputer":
            return imputers.get_IterativeImputer_config_space(n_features=n_features, random_state=random_state)
        case "KNNImputer":
            return imputers.get_KNNImputer_config_space(n_samples=n_samples)

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
   

def get_search_space(name, n_classes=3, n_samples=100, n_features=100, random_state=None, return_choice_pipeline=True, base_node=EstimatorNode):


    #if list of names, return a list of EstimatorNodes
    if isinstance(name, list) or isinstance(name, np.ndarray):
        search_spaces = [get_search_space(n, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state, return_choice_pipeline=False, base_node=base_node) for n in name]
        #remove Nones
        search_spaces = [s for s in search_spaces if s is not None]

        if return_choice_pipeline:
            return ChoicePipeline(search_spaces=np.hstack(search_spaces))
        else:
            return np.hstack(search_spaces)
    
    if name in GROUPNAMES:
        name_list = GROUPNAMES[name]
        return get_search_space(name_list, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state, return_choice_pipeline=return_choice_pipeline, base_node=base_node)
    
    return get_node(name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state, base_node=base_node)


def get_node(name, n_classes=3, n_samples=100, n_features=100, random_state=None, base_node=EstimatorNode):

    #these are wrappers that take in another estimator as a parameter
    # TODO Add AdaBoostRegressor, AdaBoostClassifier as wrappers? wrap a decision tree with different params?
    # TODO add other meta-estimators?
    if name == "RFE_classification":
        rfe_sp = get_configspace(name="RFE", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesClassifier", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(estimator_search_space=ext, method=RFE, space=rfe_sp)
    if name == "RFE_regression":
        rfe_sp = get_configspace(name="RFE", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesRegressor", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(estimator_search_space=ext, method=RFE, space=rfe_sp)
    if name == "SelectFromModel_classification":
        sfm_sp = get_configspace(name="SelectFromModel", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesClassifier", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(estimator_search_space=ext, method=SelectFromModel, space=sfm_sp)
    if name == "SelectFromModel_regression":
        sfm_sp = get_configspace(name="SelectFromModel", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        ext = get_node("ExtraTreesRegressor", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(estimator_search_space=ext, method=SelectFromModel, space=sfm_sp)
    # TODO Add IterativeImputer with more estimator methods
    '''
    if name == "IterativeImputer_learnedestimators":
        iteative_sp = get_configspace(name="IterativeImputer", n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        regessor_searchspace = get_search_space(["LinearRegression", ..], n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return WrapperPipeline(estimator_search_space=regressor_searchspace, method=ItartiveImputer, space=iteative_sp)
    '''
    #these are nodes that have special search spaces which require custom parsing of the hyperparameters
    if name == "IterativeImputer":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return EstimatorNode(STRING_TO_CLASS[name], configspace, hyperparameter_parser=imputers.IterativeImputer_hyperparameter_parser)
    if name == "RobustScaler":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=transformers.robust_scaler_hyperparameter_parser)
    if name == "GradientBoostingClassifier":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.GradientBoostingClassifier_hyperparameter_parser)
    if name == "HistGradientBoostingClassifier":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.HistGradientBoostingClassifier_hyperparameter_parser)
    if name == "GradientBoostingRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.GradientBoostingRegressor_hyperparameter_parser)
    if  name == "HistGradientBoostingRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.HistGradientBoostingRegressor_hyperparameter_parser)
    if name == "MLPClassifier":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.MLPClassifier_hyperparameter_parser)
    if name == "MLPRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.MLPRegressor_hyperparameter_parser)
    if name == "GaussianProcessRegressor":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=regressors.GaussianProcessRegressor_hyperparameter_parser)
    if name == "GaussianProcessClassifier":
        configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, random_state=random_state)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=classifiers.GaussianProcessClassifier_hyperparameter_parser)
    if name == "FeatureAgglomeration":
        configspace = get_configspace(name, n_features=n_features)
        return base_node(STRING_TO_CLASS[name], configspace, hyperparameter_parser=transformers.FeatureAgglomeration_hyperparameter_parser)

    configspace = get_configspace(name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)
    if configspace is None:
        #raise warning
        warnings.warn(f"Could not find configspace for {name}")
        return None
    
    return base_node(STRING_TO_CLASS[name], configspace)
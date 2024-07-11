from tpot2 import GraphPipeline
import numpy as np
import sklearn
import warnings
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, ElasticNetCV, PassiveAggressiveClassifier, ARDRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor,RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

    # MultinomialNB: params_MultinomialNB,

from sklearn.base import is_classifier, is_regressor

#https://scikit-learn.org/stable/auto_examples/applications/plot_model_complexity_influence.html
def _count_nonzero_coefficients_and_intercept(est):
    n_coef = np.count_nonzero(est.coef_)
    if hasattr(est, 'intercept_'):
        n_coef += np.count_nonzero(est.intercept_)

    return n_coef

#https://stackoverflow.com/questions/51139875/sklearn-randomforestregressor-number-of-trainable-parameters
def tree_complexity(tree):
    return tree.tree_.node_count * 5 #each node has 5 parameters

#https://stackoverflow.com/questions/51139875/sklearn-randomforestregressor-number-of-trainable-parameters
def forest_complexity(forest):
    all_trees = np.array(forest.estimators_)
    if len(all_trees.shape)>1:
        all_trees = all_trees.ravel()
    return sum(tree_complexity(tree) for tree in all_trees)


def histgradientboosting_complexity(forest):
    all_trees = np.array(forest._predictors)
    if len(all_trees.shape)>1:
        all_trees = all_trees.ravel()
    return sum(len(tree.nodes)*5 for tree in all_trees)

def knn_complexity(knn):
    return knn.n_neighbors


def support_vector_machine_complexity(svm):
    count = 0
    count += sum(svm.n_support_)
    if svm.kernel == 'linear':
        count += np.count_nonzero(svm.coef_)
    
    return count

def sklearn_MLP_complexity(mlp):
    n_layers = len(mlp.coefs_)
    n_params = 0
    for i in range(n_layers):
        n_params += len(mlp.coefs_[i]) + len(mlp.intercepts_[i])
    return n_params

def calculate_xgb_model_complexity(est):
    df = est.get_booster().trees_to_dataframe()
    cols_to_remove = ['Tree','Node', 'ID', 'count', 'Gain', 'Cover']
    #keeps ['Feature', 'Split', 'Yes', 'No', 'Missing', 'Category']
    #category is the specific category for a given feature. takes the place of split for categorical features

    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)

    df = ~df.isna()
    return df.sum().sum()

def BernoulliNB_Complexity(model):
    num_coefficients = len(model.class_log_prior_) + len(model.feature_log_prob_)
    return num_coefficients

def GaussianNB_Complexity(model):
    num_coefficients = len(model.class_prior_) + len(model.theta_) + len(model.var_)
    return num_coefficients

def MultinomialNB_Complexity(model):
    num_coefficients = len(model.class_log_prior_) + len(model.feature_log_prob_)
    return num_coefficients

def BaggingComplexity(est):
    return sum([calculate_model_complexity(bagged) for bagged in est.estimators_])

def lightgbm_complexity(est):
    df = est.booster_.trees_to_dataframe()
    #remove tree_index and node_depth
    cols_to_remove = ['node_index','tree_index', 'node_depth', 'count', 'parent_index']

    for col in cols_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)

    s = df.shape
    return s[0] * s[1]

def QuadraticDiscriminantAnalysis_complexity(est):
    count = reduce(operator.mul,np.array(est.rotations_).shape) + reduce(operator.mul,np.array(est.scalings_).shape) + reduce(operator.mul,np.array(est.means_).shape) + reduce(operator.mul,np.array(est.priors_).shape)
    return count

#TODO consider the complexity of the kernel?
def gaussian_process_classifier_complexity(est):
    if isinstance(est.base_estimator_, OneVsOneClassifier) or isinstance(est.base_estimator_, OneVsRestClassifier):
        count = 0
        for clf in est.base_estimator_.estimators_:
            count += len(clf.pi_)  
        return count
    return len(est.base_estimator_.pi_) 

#TODO consider the complexity of the kernel?
def gaussian_process_regressor_complexity(est):
    return len(est.alpha_)

def adaboost_complexity(est):
    return len(est.estimator_weights_) + sum(calculate_model_complexity(bagged) for bagged in est.estimators_)

def ensemble_complexity(est):
    return sum(calculate_model_complexity(bagged) for bagged in est.estimators_)


complexity_objective_per_estimator =    {   LogisticRegression: _count_nonzero_coefficients_and_intercept,
                                            SGDClassifier: _count_nonzero_coefficients_and_intercept,
                                            LinearSVC : _count_nonzero_coefficients_and_intercept,
                                            LinearSVR : _count_nonzero_coefficients_and_intercept,
                                            ARDRegression: _count_nonzero_coefficients_and_intercept, #When predicting mean, only coef and intercept used. Though there are more params for the variance/covariance matrix
                                            LinearDiscriminantAnalysis: _count_nonzero_coefficients_and_intercept,
                                            QuadraticDiscriminantAnalysis: QuadraticDiscriminantAnalysis_complexity,

                                            SGDRegressor: _count_nonzero_coefficients_and_intercept, 
                                            Ridge: _count_nonzero_coefficients_and_intercept, 
                                            Lasso: _count_nonzero_coefficients_and_intercept, 
                                            ElasticNet: _count_nonzero_coefficients_and_intercept, 
                                            Lars: _count_nonzero_coefficients_and_intercept, 
                                            LassoLars: _count_nonzero_coefficients_and_intercept, 
                                            LassoLarsCV: _count_nonzero_coefficients_and_intercept, 
                                            RidgeCV: _count_nonzero_coefficients_and_intercept, 
                                            ElasticNetCV: _count_nonzero_coefficients_and_intercept,
                                            PassiveAggressiveClassifier: _count_nonzero_coefficients_and_intercept,
                                            
                                            KNeighborsClassifier: knn_complexity,
                                            KNeighborsRegressor: knn_complexity,

                                            DecisionTreeClassifier: tree_complexity,
                                            DecisionTreeRegressor: tree_complexity,

                                            GradientBoostingRegressor: forest_complexity,
                                            GradientBoostingClassifier: forest_complexity,
                                            RandomForestClassifier : forest_complexity,
                                            RandomForestRegressor: forest_complexity,

                                            HistGradientBoostingClassifier: histgradientboosting_complexity,
                                            HistGradientBoostingRegressor: histgradientboosting_complexity,

                                            ExtraTreesRegressor: forest_complexity,
                                            ExtraTreesClassifier: forest_complexity,

                                            XGBClassifier: calculate_xgb_model_complexity,
                                            XGBRegressor: calculate_xgb_model_complexity,

                                            SVC : support_vector_machine_complexity,
                                            SVR : support_vector_machine_complexity,
                                            
                                            MLPClassifier: sklearn_MLP_complexity,
                                            MLPRegressor: sklearn_MLP_complexity,

                                            BaggingRegressor: BaggingComplexity,
                                            BaggingClassifier: BaggingComplexity,

                                            BernoulliNB: BernoulliNB_Complexity,
                                            GaussianNB: GaussianNB_Complexity,
                                            MultinomialNB: MultinomialNB_Complexity,

                                            LGBMClassifier: lightgbm_complexity,
                                            LGBMRegressor: lightgbm_complexity,

                                            GaussianProcessClassifier: gaussian_process_classifier_complexity,
                                            GaussianProcessRegressor: gaussian_process_regressor_complexity,

                                            AdaBoostClassifier: adaboost_complexity,
                                            AdaBoostRegressor: adaboost_complexity,

                                            # StackingClassifier: ensemble_complexity,
                                            # StackingRegressor: ensemble_complexity,
                                            # VotingClassifier: ensemble_complexity,
                                            # VotingRegressor: ensemble_complexity

                                        }


def calculate_model_complexity(est):
    if isinstance(est, sklearn.pipeline.Pipeline):
        return sum(calculate_model_complexity(estimator) for _,estimator in est.steps)
    if isinstance(est, sklearn.pipeline.FeatureUnion):
        return sum(calculate_model_complexity(estimator) for _,estimator in est.transformer_list)
    if isinstance(est, GraphPipeline):
        return sum(calculate_model_complexity(est.graph.nodes[node]['instance']) for node in est.graph.nodes)

    model_type = type(est)

    if is_classifier(est) or is_regressor(est):
        if model_type not in complexity_objective_per_estimator:
            warnings.warn(f"Complexity objective not defined for this classifier/regressor: {model_type}")


    if model_type in complexity_objective_per_estimator:
        return complexity_objective_per_estimator[model_type](est)
    #else, if is subclass of sklearn selector
    elif issubclass(model_type, sklearn.feature_selection.SelectorMixin):
        return 0
    else:
        return 1


def complexity_scorer(est, X, y):
    return calculate_model_complexity(est)


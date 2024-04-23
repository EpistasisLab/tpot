from tpot2 import GraphPipeline
import numpy as np
import sklearn

from tpot2.builtin_modules import genetic_encoders, feature_encoding_frequency_selector
from tpot2.builtin_modules import AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer
from tpot2.builtin_modules.genetic_encoders import DominantEncoder, RecessiveEncoder, HeterosisEncoder, UnderDominanceEncoder, OverDominanceEncoder 
from tpot2.builtin_modules import ZeroCount, ColumnOneHotEncoder
from sklearn.linear_model import SGDClassifier, LogisticRegression, SGDRegressor, Ridge, Lasso, ElasticNet, Lars, LassoLars, LassoLarsCV, RidgeCV, ElasticNetCV, PassiveAggressiveClassifier, ARDRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor,RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, PolynomialFeatures, Normalizer, MinMaxScaler, MaxAbsScaler, Binarizer
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold, RFE, SelectFromModel
from sklearn.feature_selection import f_classif, f_regression #TODO create a selectomixin using these?
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

    # MultinomialNB: params_MultinomialNB,



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

def knn_complexity(knn):
    return knn.n_neighbors


def support_vector_machine_complexity(svm):
    return sum(svm.n_support_)

def sklearn_MLP_complexity(mlp):
    n_layers = len(mlp.coefs_)
    n_params = 0
    for i in range(n_layers):
        n_params += len(mlp.coefs_[i]) + len(mlp.intercepts_[i])
    return n_params

# TODO use the complexity defined by XGBoost?
def calculate_xgb_model_complexity(model):
    num_nodes = len(model.get_booster().trees_to_dataframe())
    return num_nodes*5

def BernoulliNB_Complexity(model):
    num_coefficients = len(model.class_log_prior_) + len(model.feature_log_prob_)
    return num_coefficients

def GaussianNB_Complexity(model):
    num_coefficients = len(model.class_prior_) + len(model.theta_) + len(model.var_)
    return num_coefficients

def MultinomialNB_Complexity(model):
    num_coefficients = len(model.class_log_prior_) + len(model.feature_log_prob_)
    return num_coefficients

complexity_objective_per_estimator =    {   LogisticRegression: _count_nonzero_coefficients_and_intercept,
                                            SGDClassifier: _count_nonzero_coefficients_and_intercept,
                                            LinearSVC : _count_nonzero_coefficients_and_intercept,
                                            RandomForestClassifier : forest_complexity,
                                            
                                            KNeighborsClassifier: knn_complexity,
                                            DecisionTreeClassifier: tree_complexity,
                                            GradientBoostingClassifier: forest_complexity,
                                            ExtraTreesClassifier: forest_complexity,

                                            XGBClassifier: calculate_xgb_model_complexity,

                                            SVC : support_vector_machine_complexity,
                                            
                                            MLPClassifier: sklearn_MLP_complexity,
                                            BernoulliNB: BernoulliNB_Complexity,
                                            GaussianNB: GaussianNB_Complexity,
                                        }


def calculate_model_complexity(est):
    if isinstance(est, sklearn.pipeline.Pipeline):
        return sum(calculate_model_complexity(estimator) for _,estimator in est.steps)
    if isinstance(est, sklearn.pipeline.FeatureUnion):
        return sum(calculate_model_complexity(estimator) for _,estimator in est.transformer_list)
    if isinstance(est, GraphPipeline):
        return sum(calculate_model_complexity(est.graph.nodes[node]['instance']) for node in est.graph.nodes)

    model_type = type(est)
    if model_type in complexity_objective_per_estimator:
        return complexity_objective_per_estimator[model_type](est)
    #else, if is subclass of sklearn selector
    elif issubclass(model_type, sklearn.feature_selection.SelectorMixin):
        return 0
    else:
        return 1


def complexity_scorer(est, X, y):
    return calculate_model_complexity(est)


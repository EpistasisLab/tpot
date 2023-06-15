from tpot2 import GraphPipeline

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

from functools import partial
#import GaussianNB

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

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
from functools import partial


#TODO: how to best support transformers/selectors that take other transformers with their own hyperparameters? 
import numpy as np
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import sklearn.feature_selection
from functools import partial
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from tpot2.builtin_modules import RFE_ExtraTreesClassifier, SelectFromModel_ExtraTreesClassifier, RFE_ExtraTreesRegressor, SelectFromModel_ExtraTreesRegressor


from functools import partial
from tpot2.builtin_modules import ZeroCount, OneHotEncoder
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
                                            RandomForestClassifier : forest_complexity,
                                            SGDClassifier: _count_nonzero_coefficients_and_intercept,
                                            KNeighborsClassifier: knn_complexity,
                                            DecisionTreeClassifier: tree_complexity,
                                            GradientBoostingClassifier: forest_complexity,
                                            ExtraTreesClassifier: forest_complexity,
                                            XGBClassifier: calculate_xgb_model_complexity,
                                            SVC : support_vector_machine_complexity,
                                            LinearSVC : _count_nonzero_coefficients_and_intercept,
                                            MLPClassifier: sklearn_MLP_complexity,
                                            BernoulliNB: BernoulliNB_Complexity,
                                            GaussianNB: GaussianNB_Complexity,
                                        }


def calculate_model_complexity(est):
    if isinstance(est, sklearn.pipeline.Pipeline) or isinstance(est, sklearn.pipeline.FeatureUnion):
        return sum(calculate_model_complexity(estimator) for estimator in est.steps)
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


# coding: utf-8
from tpot.operator_utils import ARGType, TPOTOperatorClassFactory, Operator
from tpot.config_classifier import classifier_config_dict
from sklearn.base import BaseEstimator


class TPOTBase(BaseEstimator):
    """TPOT automatically creates and optimizes machine learning pipelines using genetic programming"""
    operator_dict = classifier_config_dict
    ops = []
    arglist = []
    for key in sorted(operator_dict.keys()):
        print('Creating: {}'.format(key))
        op_class, arg_types = TPOTOperatorClassFactory(key, operator_dict[key], classification=True)
        ops.append(op_class)
        arglist += arg_types

t = TPOTBase
t()

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=200, n_features=50,
                                            n_informative=10, n_redundant=10, random_state=42)

clr2 = make_pipeline(
    MinMaxScaler(),
    Normalizer(norm="l1"),
    StandardScaler(),
    MaxAbsScaler(),
    SelectFromModel(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.1), threshold=0.9500000000000001),
    LogisticRegression(C=0.0001, dual=True, penalty="l2")
)

clr2.fit(X,y)

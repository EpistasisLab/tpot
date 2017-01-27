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

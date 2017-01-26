# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

import numpy as np
from types import FunctionType

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from config_classifier import classifier_config_dict
#from config_regressor import regressor_config_dict

try:
    from inspect import signature  # Python 3
except ImportError:
    from inspect import getargspec  # Python 2

class CombineDFs(object):
    """Operator to combine two DataFrames"""

    @property
    def __name__(self):
        return self.__class__.__name__


class Operator(object):
    """Base class for operators in TPOT"""

    @property
    def __name__(self):
        """Necessary for deap so that it can generate a string identifier for
        each opeartor.
        """
        return self.__class__.sklearn_class.__name__


    @classmethod
    def inheritors(cls):
        """Returns set of all operators defined

        Parameters
        ----------
        None

        Returns
        -------
        operators: set
            Set of all discovered operators that inherit from the base class

        """
        operators = set()

        # Search two levels deep and report leaves in inheritance tree
        for operator_type in cls.__subclasses__():
            for operator in operator_type.__subclasses__():
                operators.add(operator())  # Instantiate class and append

        return operators

    @classmethod
    def get_by_name(cls, name):
        """Returns operator class instance by name

        Parameters
        ----------
        name: str
            Name of the sklearn class that belongs to a TPOT operator

        Returns
        -------
        grandchild
            An instance of the TPOT operator with a matching sklearn class name

        """
        for operator_type in cls.__subclasses__():
            for operator in operator_type.__subclasses__():
                if operator.sklearn_class.__name__ == name:
                    return operator()

class TPOTOperator(Operator):
    """
    A template of TPOT Operator Class

    """
    def __init__(self):
        pass

    root = False  # Whether this operator type can be the root of the tree
    regression = False  # Whether this operator can be used in a regression problem
    classification = False  # Whether the operator can be used for classification
    import_hash = None
    sklearn_class = None
    arg_types = None
    dep_op_list = {} # the estimator or score_func as params in this operators


class ARGType(object):
     """Base class for parameter specifications"""
     @classmethod
     def inheritors(cls):
        """Returns set of all parameter types defined

         Returns
         -------
         operators: list
            List of all discovered operators that inherit from the base class
        """
        return cls.__subclasses__()


def source_decode(sourcecode):
    """ Decode operator source and import operator class
    Parameters
    ----------
    sourcecode: string
        a string of operator source (e.g 'sklearn.feature_selection.RFE')


    Returns
    -------
    import_str: string
        a string of operator class source (e.g. 'sklearn.feature_selection')
    op_str: string
        a string of operator class (e.g. 'RFE')
    op_obj: object
        operator class (e.g. RFE)

    """
    tmp_path = sourcecode.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)
    if sourcecode.startswith('tpot.'):
        exec('from {} import {}'.format(import_str[5:], op_str)) # need update to 4:
    else:
        exec('from {} import {}'.format(import_str, op_str))
    op_obj = eval(op_str)
    return import_str, op_str, op_obj

def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    """
    Dynamically create parameter type class
    """
    return type(classname, (BaseClass,), {'values':prange})

def TPOTOperatorClassFactory(opsourse, opdict, regression=False, classification=False, BaseClass=TPOTOperator):
    """Dynamically create operator class
    Parameters
    ----------
    opsourse: string
        operator source in config dictionary (key)
    opdict: dictionary
        operator params in config dictionary (value)
    regression: bool
        True if it can be used in TPOTRegressor
    classification: bool
        True if it can be used in TPOTClassifier
    BaseClass: Class
        inherited BaseClass

    Returns
    -------
    newclass: Class
        newclass for operators
    """


    class_profile = {}
    class_profile['regression'] = regression
    class_profile['classification'] = classification

    dep_op_list = {}
    import_str, op_str, op_obj = source_decode(opsourse)
    # define if the operator can be the root of a pipeline
    if issubclass(op_obj, ClassifierMixin) or issubclass(op_obj, RegressorMixin):
        class_profile['root'] = True
        optype = "Classifier or Regressor"
    else:
        optype = "Preprocessor or Selector"

    def op_type():
        """Returns the type of the operator, e.g:
        ("Classifier", "Regressor", "Selector", "Preprocessor")
        """
        return optype

    class_profile['type'] = op_type

    class_profile['sklearn_class'] = op_obj
    import_hash = {}
    import_hash[import_str] = [op_str]
    arg_types = []
    for pname, prange in opdict.items():
        if not isinstance(prange, dict):
            classname = '{}__{}'.format(op_str, pname)
            arg_types.append(ARGTypeClassFactory(classname, prange))
        else:
            for dkey, dval in prange.items():
                dep_import_str, dep_op_str, dep_op_obj = source_decode(dkey)
                if dep_import_str in import_hash:
                    import_hash[import_str].append(dep_op_str)
                else:
                    import_hash[dep_import_str] = [dep_op_str]
                dep_op_list[pname]=dep_op_str
                if dval:
                    for dpname, dprange in dval.items():
                        classname = '{}__{}__{}'.format(op_str, dep_op_str, dpname)
                        arg_types.append(ARGTypeClassFactory(classname, dprange))
    class_profile['arg_types'] = tuple(arg_types)
    class_profile['import_hash'] = import_hash
    class_profile['dep_op_list'] = dep_op_list

    def parameter_types():
        """Return tuple of argument types for calling of the operator and the
        return type of the operator

        Parameters
        ----------
        None

        Returns
        -------
        parameter_types: tuple
            Tuple of the DEAP parameter types and the DEAP return type for the
            operator

        """
        return ([np.ndarray] + arg_types, np.ndarray)


    class_profile['parameter_types'] = parameter_types

    def export(*args):
        """Represent the operator as a string so that it can be exported to a
        file

        Parameters
        ----------
        args
            Arbitrary arguments to be passed to the operator

        Returns
        -------
        export_string: str
            String representation of the sklearn class with its parameters in
            the format:
            SklearnClassName(param1="val1", param2=val2)

        """

        op_arguments = []
        if dep_op_list:
            dep_op_arguments = {}
        for arg_class, arg_value in zip(arg_types, args):
            aname_split = arg_class.__name__.split('__')
            if isinstance(arg_value, str):
                arg_value = '\"{}\"'.format(arg_value)
            if len(aname_split) == 2: # simple parameter
                op_arguments.append("{}={}".format(aname_split[-1], arg_value))
            else:
                if not list(dep_op_list.values()).count(aname_split[1]):
                    raise TypeError('Warning: the {} is not in right format!'.format(self.sklearn_class.__name__))
                else:
                    if aname_split[1] not in dep_op_arguments:
                        dep_op_arguments[aname_split[1]] = []
                    dep_op_arguments[aname_split[1]].append("{}={}".format(aname_split[-1], arg_value))
        if dep_op_list:
            for dep_op_pname, dep_op_str in dep_op_list.items():
                if dep_op_str == 'f_classif':
                    arg_value = dep_op_str
                else:
                    arg_value = "{}({})".format(dep_op_str, ", ".join(dep_op_arguments[dep_op_str]))
                op_arguments.append("{}={}".format(dep_op_pname, arg_value))

        return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

    class_profile['export'] = export



    op_classname = '{}__{}'.format('TPOT',op_str)
    return type(op_classname, (BaseClass,), class_profile)


# for tpot
operators = Operator.inheritors()
argument_types = ARGType.inheritors()


"""
Test
op_class_dict={}

for key, val in classifier_config_dict.items():
    print('Config: {}'.format(key))
    op_class_dict[key]=TPOTOperatorClassFactory(key, val, classification=True)
    print(op_class_dict[key].sklearn_class.__name__)
    print(op_class_dict[key].import_hash)
    print(op_class_dict[key].arg_types)
a = op_class_dict['sklearn.naive_bayes.MultinomialNB']





for op in Operator.inheritors():
    print(op.sklearn_class.__name__)

for arg in ARGType.inheritors():
    print(arg.__name__, arg.values)

"""

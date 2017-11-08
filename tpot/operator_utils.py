# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
import inspect


class Operator(object):
    """Base class for operators in TPOT."""

    root = False  # Whether this operator type can be the root of the tree
    import_hash = None
    sklearn_class = None
    arg_types = None
    dep_op_list = {}  # the estimator or score_func as params in this operators


class ARGType(object):
    """Base class for parameter specifications."""

    pass


def source_decode(sourcecode):
    """Decode operator source and import operator class.

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
    try:
        if sourcecode.startswith('tpot.'):
            exec('from {} import {}'.format(import_str[4:], op_str))
        else:
            exec('from {} import {}'.format(import_str, op_str))
        op_obj = eval(op_str)
    except ImportError:
        print('Warning: {} is not available and will not be used by TPOT.'.format(sourcecode))
        op_obj = None

    return import_str, op_str, op_obj


def set_sample_weight(pipeline_steps, sample_weight=None):
    """Recursively iterates through all objects in the pipeline and sets sample weight.

    Parameters
    ----------
    pipeline_steps: array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
    sample_weight: array-like
        List of sample weight
    Returns
    -------
    sample_weight_dict:
        A dictionary of sample_weight

    """
    sample_weight_dict = {}
    if not isinstance(sample_weight, type(None)):
        for (pname, obj) in pipeline_steps:
            if inspect.getargspec(obj.fit).args.count('sample_weight'):
                step_sw = pname + '__sample_weight'
                sample_weight_dict[step_sw] = sample_weight

    if sample_weight_dict:
        return sample_weight_dict
    else:
        return None


def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    """Dynamically create parameter type class.

    Parameters
    ----------
    classname: string
        parameter name in a operator
    prange: list
        list of values for the parameter in a operator
    BaseClass: Class
        inherited BaseClass for parameter

    Returns
    -------
    Class
        parameter class

    """
    return type(classname, (BaseClass,), {'values': prange})


def TPOTOperatorClassFactory(opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType):
    """Dynamically create operator class.

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
        inherited BaseClass for operator
    ArgBaseClass: Class
        inherited BaseClass for parameter

    Returns
    -------
    op_class: Class
        a new class for a operator
    arg_types: list
        a list of parameter class

    """
    class_profile = {}
    dep_op_list = {}
    import_str, op_str, op_obj = source_decode(opsourse)

    if not op_obj:
        return None, None
    else:
        # define if the operator can be the root of a pipeline
        if issubclass(op_obj, ClassifierMixin) or issubclass(op_obj, RegressorMixin):
            class_profile['root'] = True
            optype = "Classifier or Regressor"
        else:
            optype = "Preprocessor or Selector"

        @classmethod
        def op_type(cls):
            """Return the operator type.

            Possible values:
                "Classifier", "Regressor", "Selector", "Preprocessor"
            """
            return optype

        class_profile['type'] = op_type
        class_profile['sklearn_class'] = op_obj
        import_hash = {}
        import_hash[import_str] = [op_str]
        arg_types = []

        for pname in sorted(opdict.keys()):
            prange = opdict[pname]
            if not isinstance(prange, dict):
                classname = '{}__{}'.format(op_str, pname)
                arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass))
            else:
                for dkey, dval in prange.items():
                    dep_import_str, dep_op_str, dep_op_obj = source_decode(dkey)
                    if dep_import_str in import_hash:
                        import_hash[import_str].append(dep_op_str)
                    else:
                        import_hash[dep_import_str] = [dep_op_str]
                    dep_op_list[pname] = dep_op_str
                    if dval:
                        for dpname in sorted(dval.keys()):
                            dprange = dval[dpname]
                            classname = '{}__{}__{}'.format(op_str, dep_op_str, dpname)
                            arg_types.append(ARGTypeClassFactory(classname, dprange, ArgBaseClass))
        class_profile['arg_types'] = tuple(arg_types)
        class_profile['import_hash'] = import_hash
        class_profile['dep_op_list'] = dep_op_list

        @classmethod
        def parameter_types(cls):
            """Return the argument and return types of an operator.

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

        @classmethod
        def export(cls, *args):
            """Represent the operator as a string so that it can be exported to a file.

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
                if len(aname_split) == 2:  # simple parameter
                    op_arguments.append("{}={}".format(aname_split[-1], arg_value))
                # Parameter of internal operator as a parameter in the
                # operator, usually in Selector
                else:
                    if aname_split[1] not in dep_op_arguments:
                        dep_op_arguments[aname_split[1]] = []
                    dep_op_arguments[aname_split[1]].append("{}={}".format(aname_split[-1], arg_value))

            tmp_op_args = []
            if dep_op_list:
                # To make sure the inital operators is the first parameter just
                # for better persentation
                for dep_op_pname, dep_op_str in dep_op_list.items():
                    if dep_op_pname == 'score_func':
                        arg_value = dep_op_str
                    else:
                        arg_value = "{}({})".format(dep_op_str, ", ".join(dep_op_arguments[dep_op_str]))
                    tmp_op_args.append("{}={}".format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

        class_profile['export'] = export

        op_classname = 'TPOT_{}'.format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        return op_class, arg_types

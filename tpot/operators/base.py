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
from tpot.indices import CLASS_COL, GROUP_COL, TRAINING_GROUP, non_feature_columns

try:
    from inspect import signature  # Python 3
except ImportError:
    from inspect import getargspec  # Python 2


class Operator(object):
    """Base class for operators in TPOT"""

    # Default parameters for sklearn classes
    default_arguments = {
        'random_state': 42,
        'n_jobs': -1
    }

    def __call__(self, input_matrix, *args, **kwargs):
        input_matrix = np.copy(input_matrix)  # Make a copy of the input dataframe

        self.training_features = input_matrix[input_matrix[:, GROUP_COL] == TRAINING_GROUP]
        np.delete(self.training_features, non_feature_columns, axis=1)
        self.training_classes = input_matrix[input_matrix[:, GROUP_COL] == TRAINING_GROUP][:, CLASS_COL]

        # If there are no features left then there is nothing to do
        if self.training_features.shape[1] == 0:
            return input_matrix

        # Call child class' _call function
        return self._call(input_matrix, *args, **kwargs)

    def export(self, *args, **kwargs):
        """Represent the operator as a string so that it can be exported to a
        file

        Parameters
        ----------
        args, kwargs
            Arbitrary arguments to be passed to the operator

        Returns
        -------
        export_string: str
            String representation of the sklearn class with its parameters in
            the format:
            SklearnClassName(param1="val1", param2=val2)

        """
        operator_args = self.preprocess_args(*args, **kwargs)

        arguments = []
        for key in sorted(operator_args.keys()):
            val = operator_args[key]
            if isinstance(val, str):
                val = '\"{}\"'.format(val)
            elif isinstance(val, FunctionType):
                val = val.__name__

            arguments.append("{}={}".format(key, val))

        return "{}({})".format(self.sklearn_class.__name__, ", ".join(arguments))

    @property
    def __name__(self):
        """Necessary for deap so that it can generate a string identifier for
        each opeartor.
        """
        return self.__class__.sklearn_class.__name__

    @property
    def type(self):
        """Returns the type of the operator, e.g:
        ("Classifier", "Selector", "Preprocessor")
        """
        return self.__class__.__bases__[0].__name__

    def _merge_with_default_params(self, kwargs):
        """Apply defined default parameters to the sklearn class where applicable
        while also integrating specified arguments

        Parameters
        ----------
        kwargs: dict
            Preprocessed arguments from DEAP

        Returns
        -------
        sklearn_class
            The class's sklearn_class instantiated with both the default
            arguments and specified arguments.

        """
        try:
            # Python 3
            sklearn_argument_names = set(signature(self.sklearn_class).
                parameters.keys())
        except NameError:
            # Python 2
            try:
                sklearn_argument_names = \
                    set(getargspec(self.sklearn_class.__init__).args)
            except TypeError:
                # For when __init__ comes from C code and can not be inspected
                sklearn_argument_names = set()  # Assume no parameters

        default_argument_names = set(self.default_arguments.keys())

        # Find which arguments are defined in both the defaults and the
        # sklearn class
        applicable_defaults = sklearn_argument_names.\
            intersection(default_argument_names)

        for new_arg in applicable_defaults:
            kwargs[new_arg] = self.default_arguments[new_arg]

        return self.sklearn_class(**kwargs)

    def parameter_types(self):
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
        try:
            # Python 3
            num_args = len(signature(self.preprocess_args).parameters.keys())
        except NameError:
            # Python 2

            # Remove 'self'
            num_args = len(getargspec(self.preprocess_args).args[1:])

        # Make sure the class has been written properly
        if num_args != len(self.arg_types):
            raise RuntimeError(("{}'s arg_types does not correspond to the "
                                "arguments defined for itself".
                                format(self.__name__)))

        # First argument is always a DataFrame
        arg_types = [np.ndarray] + list(self.arg_types)
        return_type = np.ndarray

        return (arg_types, return_type)

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

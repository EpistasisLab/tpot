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

try:
    from inspect import signature  # Python 3
except ImportError:
    from inspect import getargspec  # Python 2


class Operator(object):
    """Base class for operators in TPOT"""

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
        ("Classifier", "Regressor", "Selector", "Preprocessor")
        """
        return self.__class__.__bases__[0].__name__

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

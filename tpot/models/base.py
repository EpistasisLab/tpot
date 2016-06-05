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
details.
You should have received a copy of the GNU General Public License along with
the TPOT library. If not, see http://www.gnu.org/licenses/.
"""

from traitlets import (
    HasTraits,
)
from pandas import (
    DataFrame,
    Series,
)
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.feature_selection import (
    base,
)
from toolz import (
    partial, pipe, identity,
)

default_class_column = 'species'


def evaluate_classifier(model, X):
    v = model.predict(
        X.values,
    )
    return Series(
        v, name='guess',
        index=X.index,
    )


def evaluate_selection(model, X):
    return X[model.get_support(True)]


def evaluate_transform(model, X):
    try:
        transformed = model.transform(
            X.values, X.index.get_level_values(-1)
        )
    except:
        transformed = model.transform(X.values)
    return DataFrame(
        transformed,
        index=X.index,
    )

class EvaluateEstimator(HasTraits):
    output_type = DataFrame
    default_params = {}

    @property
    def exports(self):
        for subclass in [
            base.SelectorMixin,
            RegressorMixin,
            ClassifierMixin,
            TransformerMixin,
        ]:
            if issubclass(self.model, subclass):
                return subclass

    @classmethod
    def trait_types(cls):
        instance = cls()
        params = instance.model._get_param_names()
        traits = instance.trait_names()
        return [
            type(getattr(instance, key)) for key in params if key in traits
        ]

    def _args_to_kwargs(self, X, *args):
        traits = self.trait_names()
        kwargs = {}
        if args:
            kwargs = {
                k: args[i] for i, k in enumerate(traits)
            }
        for key, value in kwargs.items():
            func = self.trait_metadata(key, 'apply')
            if func:
                if self.trait_metadata(key, 'df'):
                    func = partial(func, X)
            else:
                func = identity
            kwargs[key] = func(value)
        return kwargs

    @classmethod
    def fit_primitive(cls, X, *args):
        self = cls()

        kwargs = self._args_to_kwargs(X, *args)
        
        # Initialize the model in the class
        model = self.model(
            **kwargs,
        )

        # Apply any default parameters to the model.
        model.set_params(**cls.default_params)

        # Fit the model
        if not isinstance(X, DataFrame):
            X = DataFrame(X)

        if len(X.columns) == 0:
            return X

        # Fit the testing data
        model.fit(X.ix[True].values, X.ix[True].index.values.ravel())

        # Use sklearn Mixins to choose the output
        return self.mixins(model, X)

    def mixins(self, model, X):
        method = {
            base.SelectorMixin: evaluate_selection,
            RegressorMixin: evaluate_classifier,
            ClassifierMixin: evaluate_classifier,
            TransformerMixin: evaluate_transform,
        }[self.exports]
        return method(model, X)


class PredictEstimator(EvaluateEstimator):
    output_type = Series

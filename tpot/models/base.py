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
)
from sklearn.base import (
    BaseEstimator,
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


class PipelineEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, model=identity, class_column=default_class_column):
        self.model = model
        self.class_column = class_column

    def predict(self, X):
        result = self.model(X).ix[False]
        if len(result.columns) == 1 and 'guess' in result.columns:
            if result.guess.isnull().values.any():
                return result['guess'].values - 1
            return result['guess'].values
        else:
            return result.index.values - 1

    def score(self, X):
        return super().score(
            X, X.ix[False].index.values
        )


class EvaluateEstimator(HasTraits):
    output_type = DataFrame
    default_params = {}

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
        kwargs = {
            k: args[i] for i, k in pipe(traits, enumerate)
        }
        for trait in traits:
            func = self.trait_metadata(trait, 'apply')
            if func:
                if self.trait_metadata(trait, 'df'):
                    func = partial(func, X)
            else:
                func = identity
            kwargs[trait] = func(kwargs[trait])
        return kwargs

    @classmethod
    def fit_predict(cls, X, *args):
        self = cls()

        # Initialize the model in the class
        model = self.model(
            **self._args_to_kwargs(X, *args)
        )

        # Apply any default parameters to the model.
        model.set_params(**cls.default_params)

        # Fit the model
        model.fit(X.ix[True].values, X.ix[True].index.values)

        # Use sklearn Mixins to choose the output
        if issubclass(model.__class__, base.SelectorMixin):
            v = X[model.get_support(True)]
            return v

        if issubclass(model.__class__, (RegressorMixin, ClassifierMixin,)):
            return DataFrame(
                    model.predict(
                        X.values,
                    ), columns=['guess'],
                    index=X.index,
                )
        if issubclass(model.__class__, TransformerMixin):
            return DataFrame(
                model.transform(
                    X.values, X.index.get_level_values(-1)
                ), index=X.index
            )

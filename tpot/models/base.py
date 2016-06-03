from traitlets import (
    HasTraits,
)
from pandas import (
    DataFrame,
)
from sklearn.base import (
    BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin,
)
from toolz import (
    partial, pipe, identity,
)


class PipelineEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, model=identity):
        self.model_ = model

    def predict(self, X):
        return self.model_(X).values

    def score(self, X):
        return super().score(
            X, X.index.values
        )


class EvaluateEstimator(HasTraits):
    output_type = DataFrame

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
        model = self.model(
            **self._args_to_kwargs(X, *args)
        )
        model.fit(X.ix[True].values, X.ix[True].index.values)
        if issubclass(model.__class__, (RegressorMixin, ClassifierMixin,)):
            return DataFrame(
                    model.predict(
                        X.values,
                    ), columns=['guess'],
                    index=X.index,
                )
        elif issubclass(model.__class__, TransformerMixin):
            return DataFrame(
                model.transform(
                    X.values, X.index.get_level_values(-1)
                ), index=X.index
            )

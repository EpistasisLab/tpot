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
    partial, pipe, identity, flip, curry
)


class PipelineEstimator(ClassifierMixin, BaseEstimator):
    @classmethod
    def compose(cls, model, *args):
        model = flip(curry(model))
        for arg in args:
            model = model(arg)
        return cls(model)

    def __init__(self, model=identity):
        self.model_ = model

    def predict(self, X):
        return self.model_(X)

    def score(self, X):
        return super().score(
            X.values, X.index
        )


class EvaluateEstimator(HasTraits):
    output_type = DataFrame

    @classmethod
    def trait_types(cls):
        instance = cls()
        params = instance.get_params()
        return [
            type(params[key]) for key in instance.trait_names()
        ]

    @classmethod
    def evaluate(cls, X, *args):
        instance = cls()
        traits = instance.trait_names()
        kwargs = {
            k: args[i] for i, k in pipe(traits, enumerate)
        }
        for trait in traits:
            func = instance.trait_metadata(trait, 'apply')
            if func:
                if instance.trait_metadata(trait, 'df'):
                    func = partial(func, X)
            else:
                func = identity
            kwargs[trait] = func(kwargs[trait])
        model = instance.model(**kwargs)
        model.fit(X.ix[True].values, X.ix[True].index.values)
        return instance.predict(cls, model, X)

    def predict(self, cls, instance, X):
        if issubclass(cls.model, (RegressorMixin, ClassifierMixin,)):
            return DataFrame(
                    instance.predict(
                        X.values,
                    ), columns=['guess'],
                    index=X.index,
                )
        elif issubclass(cls.model, TransformerMixin):
            return DataFrame(
                instance.transform(
                    X.values, X.index.get_level_values(1)
                ), index=X.index
            )

from .base import (
    EvaluateEstimator,
)
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
)
from traitlets import (
    Int, Float,
)
from toolz import (
    pipe, partial,
)


class random_forest(EvaluateEstimator):
    model = RandomForestClassifier
    n_estimators = Int(default_value=500).tag(
        apply=lambda ne: pipe(ne, partial(min, 500),  partial(max, 1))
    )
    max_features = Int(default_value=0).tag(
        df=True,
        apply=lambda df, mf:
            'auto' if mf < 1 else None if mf == 1 else len(df.columns)
    )


class ada_boost(EvaluateEstimator):
    model = AdaBoostClassifier
    n_estimators = Int().tag(
        apply=lambda ne: pipe(ne, partial(min, 500),  partial(max, 1))
    )
    learning_rate = Float().tag(
        apply=partial(max, .0001)
    )

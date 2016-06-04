from .base import (
    EvaluateEstimator,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
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
    n_estimators = Int(500).tag(
        apply=lambda ne: pipe(ne, partial(min, 500),  partial(max, 1), int)
    )
    learning_rate = Float().tag(
        apply=partial(max, .0001)
    )


class extra_trees(EvaluateEstimator):
    model = ExtraTreesClassifier
    n_estimators = Int(500).tag(
        df=True,
        apply=lambda df, mf:
            pipe(mf, partial(min, 1), partial(max, len(df.columns)))
    )
    criterion = Int(0).tag(
        apply=lambda x: ['gini', 'entropy'][x != 0]
    )
    max_features = Int(default_value=0).tag(
        df=True,
        apply=lambda df, mf:
            pipe(mf, partial(min, 1), partial(max, len(df.columns)))
    )

class gradient_boost(EvaluateEstimator):
    model = GradientBoostingClassifier

    learning_rate = Float().tag(
        apply=partial(max, 0.0001)
    )
    max_depth = Float().tag(
        apply=partial(max, 1)
    )

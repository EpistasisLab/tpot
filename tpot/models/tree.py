from .base import (
    PredictEstimator,
)
from sklearn.tree import (
    DecisionTreeClassifier,
)
from traitlets import (
    Int, Float,
)
from toolz import (
    pipe, partial,
)


class decision_tree(PredictEstimator):
    model = DecisionTreeClassifier
    max_depth = Int(default_value=500).tag(
        apply=lambda x: None if x < 1 else x,
    )
    max_features = Int(default_value=0).tag(
        df=True,
        apply=lambda df, mf:
            'auto' if mf < 1 else None if mf == 1 else len(df.columns)
    )

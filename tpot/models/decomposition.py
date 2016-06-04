from .base import (
    EvaluateEstimator,
)
from sklearn.decomposition import (
    FastICA,
)
from traitlets import (
    Int, Float,
)
from toolz import (
    partial, compose
)


class fast_ica(EvaluateEstimator):
    model = FastICA
    n_components = Int(default_value=0).tag(
        df=True,
        apply=compose(
            int,
            lambda df, nc: 1 if nc < 1 else min(nc, len(df.columns))
        )
    )
    tol = Float().tag(
        apply=partial(max, .0001)
    )

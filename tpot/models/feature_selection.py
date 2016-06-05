from .base import (
    EvaluateEstimator,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    SelectPercentile,
    RFE,
    SelectFwe,
    f_classif
)
from sklearn.svm import (
    SVC,
)
from traitlets import (
    Int, Float,
)
from toolz import (
    pipe, partial, compose
)


class variance_threshold(EvaluateEstimator):
    """Uses scikit-learn's VarianceThreshold feature selection to learn the
    subset of features that pass the threshold
    Parameters
    ----------
    threshold: float
        The variance threshold that removes features that fall under
        the threshold
    """
    model = VarianceThreshold
    threshold = Float(.5)


class select_kbest(EvaluateEstimator):
    """Uses scikit-learn's SelectKBest feature selection to learn the subset
    of features that have the highest score according to some scoring function
    Parameters
    ----------
    k: int
        The top k features to keep from the original set of features in the
        training data
    """
    model = SelectKBest
    default_params = {
        'score_func': f_classif,
    }
    k = Int(1).tag(
        df=True,
        apply=lambda df, k: pipe(
            k, partial(max, 0), partial(min, len(df.columns)),
        ),
    )


class select_percentile(EvaluateEstimator):
    """Uses scikit-learn's SelectKBest feature selection to learn the subset
    of features that have the highest score according to some scoring function
    Parameters
    ----------
    k: int
        The top k features to keep from the original set of features in the
        training data
    """
    model = SelectPercentile
    percentage = Float(50.).tag(
        apply=compose(partial(min, 100), partial(max, 0))
    )


class rfe(EvaluateEstimator):
    """Uses scikit-learn's SelectKBest feature selection to learn the subset
    of features that have the highest score according to some scoring function
    Parameters
    ----------
    k: int
        The top k features to keep from the original set of features in the
        training data
    """
    model = RFE
    default_params = {
        'estimator': SVC(kernel='linear'),
    }
    n_features_to_select = Int(50).tag(
        df=True,
        apply=lambda s, n: pipe(
            n, partial(max, 1), partial(min, len(s))
        )
    )
    step = Float(0.5).tag(
        apply=compose(partial(min, .99), partial(max, .1))
    )


class select_fwe(EvaluateEstimator):
    """Uses scikit-learn's SelectKBest feature selection to learn the subset
    of features that have the highest score according to some scoring function
    Parameters
    ----------
    k: int
        The top k features to keep from the original set of features in the
        training data
    """
    model = SelectFwe
    alpha = Float(0.5).tag(
        apply=compose(partial(min, .05), partial(max, .001))
    )

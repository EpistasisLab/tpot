from .base import (
    EvaluateEstimator,
)
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    Binarizer,
)
from traitlets import (
    Float,
)


class standard_scaler(EvaluateEstimator):
    """Uses scikit-learn's StandardScaler to scale the features by removing
    their mean and scaling to unit variance.
    """
    model = StandardScaler


class robust_scaler(EvaluateEstimator):
    """Uses scikit-learn's RobustScaler to scale the features using
    statistics that are robust to outliers
    """
    model = RobustScaler


class max_abs_scaler(EvaluateEstimator):
    """Uses scikit-learn's MaxAbsScaler to transform all of the
    features by scaling them to [0, 1] relative to the feature
    """
    model = MaxAbsScaler


class min_max_scaler(EvaluateEstimator):
    """Uses scikit-learn's MinMaxScaler to transform all of the
    features by scaling them to the range [0, 1]
    """
    model = MinMaxScaler


class polynomial_features(EvaluateEstimator):
    """Uses scikit-learn's PolynomialFeatures to construct new degree-2
    polynomial features from the existing feature set
    """
    default_params = dict(
        degree=2, include_bias=False, interaction_only=False,
    )
    model = PolynomialFeatures


class binarizer(EvaluateEstimator):
    """Uses scikit-learn's Binarizer to binarize all of the features, setting
    any feature >`threshold` to 1 and all others to 0
    """
    model = Binarizer
    threshold = Float()

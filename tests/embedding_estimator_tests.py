from sklearn.datasets import make_classification, make_regression
from tpot.builtins import EmbeddingEstimator
from sklearn.neural_network import MLPClassifier, MLPRegressor


def test_EmbeddingClassifier_1():
    """Assert that Embedding for classification works as expected."""
    X, y = make_classification(random_state=1)
    cs = EmbeddingEstimator(MLPClassifier(random_state=1, tol=0.9))
    X_transformed = cs.fit_transform(X, y)

    # 20 features + 100 embedding size
    assert X_transformed.shape[1] == 120


def test_EmbeddingClassifier_2():
    """Assert that correct embedding layer is selected (classifier)."""
    X, y = make_classification(random_state=1)
    cs = EmbeddingEstimator(
        MLPClassifier(hidden_layer_sizes=[20, 10], random_state=1, tol=0.9)
    )
    cs_2 = EmbeddingEstimator(
        MLPClassifier(hidden_layer_sizes=[20, 10], random_state=1, tol=0.9),
        embedding_layer=1,
    )
    X_transformed = cs.fit_transform(X, y)
    X_transformed_2 = cs_2.fit_transform(X, y)

    assert X_transformed.shape[1] == 30  # 20 features + 20 embedding size
    assert X_transformed_2.shape[1] == 40  # 20 features + 20 embedding size


def test_EmbeddingRegressor_1():
    """Assert that Embedding for regressor works as expected."""
    X, y = make_regression(n_features=20, random_state=1)
    cs = EmbeddingEstimator(MLPRegressor(random_state=1, tol=1000))
    X_transformed = cs.fit_transform(X, y)

    # 20 features + 100 embedding size
    assert X_transformed.shape[1] == 120


def test_EmbeddingRegressor_2():
    """Assert that correct embedding layer is selected (regressor)."""
    X, y = make_regression(n_features=20, random_state=1)
    cs = EmbeddingEstimator(
        MLPRegressor(hidden_layer_sizes=[20, 10], random_state=1, tol=1000)
    )
    cs_2 = EmbeddingEstimator(
        MLPRegressor(hidden_layer_sizes=[20, 10], random_state=1, tol=1000),
        embedding_layer=1,
    )
    X_transformed = cs.fit_transform(X, y)
    X_transformed_2 = cs_2.fit_transform(X, y)

    assert X_transformed.shape[1] == 30  # 20 features + 20 embedding size
    assert X_transformed_2.shape[1] == 40  # 20 features + 20 embedding size


def test_EmbeddingKeras():
    """Check that this works also for keras models"""
    try:
        import tensorflow as tf
    except ImportError:
        tf = None
    if tf is None:
        return
    from tensorflow.keras import backend as K
    import tensorflow.keras as keras
    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation

    def make_model(input_shape):
        model = Sequential()
        model.add(Dense(20, activation="relu", input_dim=input_shape))
        model.add(Dense(15, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    X, y = make_classification(random_state=1)
    cs = EmbeddingEstimator(KerasClassifier(make_model), backend=K)
    cs_2 = EmbeddingEstimator(
        KerasClassifier(make_model), embedding_layer=-3, backend=K
    )
    X_transformed = cs.fit_transform(X, y, verbose=0)
    X_transformed_2 = cs_2.fit_transform(X, y, verbose=0)

    assert X_transformed.shape[1] == 35  # 20 features + 15 embedding size
    assert X_transformed_2.shape[1] == 40  # 20 features + 20 embedding size

import nose
from sklearn.datasets import make_classification, make_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from tpot.builtins import EmbeddingEstimator


try:
    from tensorflow.keras import backend as K
    import tensorflow.keras as keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Activation
except ImportError:
    raise nose.SkipTest()


def _build_keras_classifier(ly_sizes, input_shape):
    assert len(ly_sizes) >= 2
    model = Sequential()
    model.add(Dense(ly_sizes[0], activation="relu", input_dim=input_shape))
    for ly_size in ly_sizes[1:-2]:
        model.add(Dense(ly_size, activation="relu"))
    model.add(Dense(ly_sizes[-1], activation="softmax"))
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def test_Embedding_Keras():
    """Assert that Embedding for classification works as expected."""
    layer_sizes = [20, 100, 50, 20, 60, 10]
    X, y = make_classification(random_state=1)
    model = _build_keras_classifier(layer_sizes, X.shape[1])

    def check(X, X_transformed, embedding_layer_size):
        assert X.shape[1] + embedding_layer_size == X_transformed.shape[1]

    for i in range(len(layer_sizes)):
        cs = EmbeddingEstimator(model, embedding_layer=i)
        X_transformed = cs.fit_transform(X, y)
        yield check, X, X_transformed, layer_sizes[i]

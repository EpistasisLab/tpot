import nose
from sklearn.datasets import make_classification, make_regression
from sklearn.neural_network import MLPClassifier, MLPRegressor

from tpot.builtins import DeepLearningTransformer

from tpot.builtins.deep_learning import HAS_TENSORFLOW


if not HAS_TENSORFLOW:
    raise nose.SkipTest()


def test_Embedding_Keras():
    """Assert that Embedding for classification works as expecated."""
    layer_sizes = [20, 100, 50, 20, 60, 100]
    X, y = make_classification(random_state=1)

    def check(X, X_transformed, embedding_layer_size):
        assert X.shape[1] + embedding_layer_size == X_transformed.shape[1]

    for i in range(len(layer_sizes) - 1):
        cs = DeepLearningTransformer(embedding_layer=i)
        X_transformed = cs.fit_transform(X=X, y=y)
        yield check, X, X_transformed, layer_sizes[i]



layer_sizes = [20, 100, 50, 20, 60, 100]
X, y = make_classification(random_state=1)

def check(X, X_transformed, embedding_layer_size):
    assert X.shape[1] + embedding_layer_size == X_transformed.shape[1]

for i in range(len(layer_sizes) - 1):
    cs = DeepLearningTransformer(hidden_layer_sizes=layer_sizes, embedding_layer=i)
    X_transformed = cs.fit_transform(X=X, y=y)
    check(X, X_transformed, layer_sizes[i])

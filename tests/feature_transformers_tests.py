from sklearn.datasets import load_iris
from tpot.builtins.feature_transformers import CategoricalSelector
from nose.tools import assert_equal, assert_raises

iris_data = load_iris().data

def test_CategoricalSelector():
    """Assert that CategoricalSelector works as expected."""
    cs = CategoricalSelector()
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],2)


def test_CategoricalSelector_2():
    """Assert that CategoricalSelector works as expected with threshold=5"""
    cs = CategoricalSelector(threshold=5)
    X_transformed = cs.transform(iris_data[0:16, :])
    
    assert_equal(X_transformed.shape[1],1)


def test_CategoricalSelector_3():
    """Assert that CategoricalSelector works as expected with threshold=20"""
    cs = CategoricalSelector(threshold=20)
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],7)


def test_CategoricalSelector_4():
    """Assert that CategoricalSelector rasies ValueError without categorical features"""
    cs = CategoricalSelector()

    assert_raises(ValueError, cs.transform, iris_data)

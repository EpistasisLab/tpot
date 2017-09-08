from sklearn.datasets import load_iris
from tpot.builtins import CategoricalSelector, ContinuousSelector
from nose.tools import assert_equal, assert_raises

iris_data = load_iris().data

def test_CategoricalSelector():
    """Assert that CategoricalSelector works as expected."""
    cs = CategoricalSelector()
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],2)


def test_CategoricalSelector_2():
    """Assert that CategoricalSelector works as expected with threshold=5."""
    cs = CategoricalSelector(threshold=5)
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],1)


def test_CategoricalSelector_3():
    """Assert that CategoricalSelector works as expected with threshold=20."""
    cs = CategoricalSelector(threshold=20)
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],7)


def test_CategoricalSelector_4():
    """Assert that CategoricalSelector rasies ValueError without categorical features."""
    cs = CategoricalSelector()

    assert_raises(ValueError, cs.transform, iris_data)


def test_CategoricalSelector_fit():
    """Assert that fit() in CategoricalSelector does nothing."""
    op = CategoricalSelector()
    ret_op = op.fit(iris_data)

    assert ret_op==op


def test_ContinuousSelector():
    """Assert that ContinuousSelector works as expected."""
    cs = ContinuousSelector(svd_solver='randomized')
    X_transformed = cs.transform(iris_data[0:16, :])

    assert_equal(X_transformed.shape[1],2)


def test_ContinuousSelector_2():
    """Assert that ContinuousSelector works as expected with threshold=5."""
    cs = ContinuousSelector(threshold=5, svd_solver='randomized')
    X_transformed = cs.transform(iris_data[0:16, :])
    assert_equal(X_transformed.shape[1],3)


def test_ContinuousSelector_3():
    """Assert that ContinuousSelector works as expected with svd_solver='full'"""
    cs = ContinuousSelector(threshold=10, svd_solver='full')
    X_transformed = cs.transform(iris_data[0:16, :])
    assert_equal(X_transformed.shape[1],2)


def test_ContinuousSelector_4():
    """Assert that ContinuousSelector rasies ValueError without categorical features."""
    cs = ContinuousSelector()

    assert_raises(ValueError, cs.transform, iris_data[0:10,:])


def test_ContinuousSelector_fit():
    """Assert that fit() in ContinuousSelector does nothing."""
    op = ContinuousSelector()
    ret_op = op.fit(iris_data)

    assert ret_op==op

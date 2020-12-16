import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.compose
from tpot.builtins import CategoricalSelector, ContinuousSelector, ColumnTransformer
from nose.tools import assert_equal, assert_raises, assert_true

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


def _make_col_transformer_kwargs(choice, transformers, cols, remainder):
    kwargs = {}
    for i, t in enumerate(transformers):
        kwargs['transformer_{}'.format(i)] = t
    for c in cols:
        kwargs['include_col_{}'.format(c)] = True
    kwargs['choice'] = choice
    kwargs['remainder'] = remainder
    return kwargs


def test_ColumnTransformer():
    """Assert ColumnTransformer matches its sklearn counterpart."""
    cols = list(range(0, 3))
    kwargs = _make_col_transformer_kwargs(0, [StandardScaler(), MinMaxScaler()], cols, 'drop')
    ct = ColumnTransformer(**kwargs)
    sklearn_ct = sklearn.compose.ColumnTransformer([("rand_name", StandardScaler(), cols)], remainder='drop')
    ct.fit(iris_data)
    sklearn_ct.fit(iris_data)
    X1 = ct.transform(iris_data)
    X2 = sklearn_ct.transform(iris_data)
    assert_true(np.array_equal(X1, X2))


def test_ColumnTransformer_2():
    """Assert that ColumnTransformer matches its sklearn counterpart with different params."""
    cols = list(range(1, 4))
    kwargs = _make_col_transformer_kwargs(1, [StandardScaler(), MinMaxScaler()], cols, 'passthrough')
    ct = ColumnTransformer(**kwargs)
    sklearn_ct = sklearn.compose.ColumnTransformer([("rand_name", MinMaxScaler(), cols)], remainder='passthrough')
    ct.fit(iris_data)
    sklearn_ct.fit(iris_data)
    X1 = ct.transform(iris_data)
    X2 = sklearn_ct.transform(iris_data)
    assert_true(np.array_equal(X1, X2))


def test_ColumnTransformer_3():
    """Assert that ColumnTransformer does nothing if bad kwargs."""
    ct = ColumnTransformer(invalid_param=True)
    ct.fit(iris_data)
    X_transformed = ct.transform(iris_data)
    assert_true(np.array_equal(X_transformed, iris_data))

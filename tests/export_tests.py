# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

from tqdm import tqdm
import numpy as np
from os import remove, path

from tpot import TPOTClassifier, TPOTRegressor
from tpot.export_utils import export_pipeline, generate_import_code, _indent, \
    generate_pipeline_code, get_by_name, set_param_recursive
from tpot.operator_utils import TPOTOperatorClassFactory
from tpot.config.classifier import classifier_config_dict

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from deap import creator

from nose.tools import assert_raises, assert_equal, nottest
train_test_split = nottest(train_test_split)
test_operator_key_1 = 'sklearn.feature_selection.SelectPercentile'
test_operator_key_2 = 'sklearn.feature_selection.SelectFromModel'
TPOTSelectPercentile, TPOTSelectPercentile_args = TPOTOperatorClassFactory(
    test_operator_key_1,
    classifier_config_dict[test_operator_key_1]
)

TPOTSelectFromModel, TPOTSelectFromModel_args = TPOTOperatorClassFactory(
    test_operator_key_2,
    classifier_config_dict[test_operator_key_2]
)

digits_data = load_digits()
training_features, testing_features, training_target, testing_target = \
    train_test_split(digits_data.data.astype(np.float64), digits_data.target.astype(np.float64), random_state=42)

tpot_obj = TPOTClassifier()
tpot_obj._fit_init()

tpot_obj_reg = TPOTRegressor()
tpot_obj_reg._fit_init()

def test_export_random_ind():
    """Assert that the TPOTClassifier can generate the same pipeline export with random seed of 39."""
    tpot_obj = TPOTClassifier(random_state=39, config_dict="TPOT light")
    tpot_obj._fit_init()
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline = tpot_obj._toolbox.individual()
    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=39)

exported_pipeline = BernoulliNB(alpha=1.0, fit_prior=False)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 39)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    exported_code = export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset, random_state=tpot_obj.random_state)
    assert expected_code == exported_code


def test_export():
    """Assert that TPOT's export function throws a RuntimeError when no optimized pipeline exists."""
    assert_raises(RuntimeError, tpot_obj.export, "test_export.py")
    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5), ZeroCount(input_matrix))'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj._optimized_pipeline = pipeline
    tpot_obj.export("test_export.py")
    assert path.isfile("test_export.py")
    remove("test_export.py") # clean up exported file


def test_export_2():
    """Assert that TPOT's export function returns the expected pipeline text as a string."""

    pipeline_string = (
        'KNeighborsClassifier('
        'input_matrix, '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    tpot_obj._optimized_pipeline = pipeline
    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == tpot_obj.export()


def test_generate_pipeline_code():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline."""

    tpot_obj._fit_init()
    pipeline = [
        'KNeighborsClassifier',
        [
            'CombineDFs',
            [
                'GradientBoostingClassifier',
                'input_matrix',
                38.0,
                5,
                5,
                5,
                0.05,
                0.5],
            [
                'GaussianNB',
                [
                    'ZeroCount',
                    'input_matrix'
                ]
            ]
        ],
        18,
        'uniform',
        2
    ]

    expected_code = """make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=38.0, max_depth=5, max_features=5, min_samples_leaf=5, min_samples_split=0.05, n_estimators=0.5)),
        StackingEstimator(estimator=make_pipeline(
            ZeroCount(),
            GaussianNB()
        ))
    ),
    KNeighborsClassifier(n_neighbors=18, p="uniform", weights=2)
)"""
    assert expected_code == generate_pipeline_code(pipeline, tpot_obj.operators)


def test_generate_pipeline_code_2():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline with two CombineDFs."""

    pipeline = [
        'KNeighborsClassifier',
        [
            'CombineDFs',
            [
                'GradientBoostingClassifier',
                'input_matrix',
                38.0,
                5,
                5,
                5,
                0.05,
                0.5],
            [
                'CombineDFs',
                [
                    'MinMaxScaler',
                    'input_matrix'
                ],
                ['ZeroCount',
                    [
                        'MaxAbsScaler',
                        'input_matrix'
                    ]
                ]
            ]
        ],
        18,
        'uniform',
        2
    ]

    expected_code = """make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=38.0, max_depth=5, max_features=5, min_samples_leaf=5, min_samples_split=0.05, n_estimators=0.5)),
        make_union(
            MinMaxScaler(),
            make_pipeline(
                MaxAbsScaler(),
                ZeroCount()
            )
        )
    ),
    KNeighborsClassifier(n_neighbors=18, p="uniform", weights=2)
)"""

    assert expected_code == generate_pipeline_code(pipeline, tpot_obj.operators)


def test_generate_import_code():
    """Assert that generate_import_code() returns the correct set of dependancies for a given pipeline."""

    pipeline = creator.Individual.from_string('GaussianNB(RobustScaler(input_matrix))', tpot_obj._pset)

    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
"""
    assert expected_code == generate_import_code(pipeline, tpot_obj.operators)


def test_generate_import_code_2():
    """Assert that generate_import_code() returns the correct set of dependancies and dependancies are importable."""

    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5), ZeroCount(input_matrix))'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    import_code = generate_import_code(pipeline, tpot_obj.operators)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
"""
    exec(import_code)  # should not raise error
    assert expected_code == import_code


def test_operators():
    """Assert that the TPOT operators match the output of their sklearn counterparts."""
    for op in tpot_obj.operators:
        check_export.description = ("Assert that the TPOT {} operator exports "
                                    "as expected".format(op.__name__))
        yield check_export, op, tpot_obj


def check_export(op, tpot_obj):
    """Assert that a TPOT operator exports as a class constructor."""
    prng = np.random.RandomState(42)
    np.random.seed(42)

    args = []
    for type_ in op.parameter_types()[0][1:]:
        args.append(prng.choice(tpot_obj._pset.terminals[type_]).value)
    export_string = op.export(*args)

    assert export_string.startswith(op.__name__ + "(") and export_string.endswith(")")


def test_export_pipeline():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed pipeline."""

    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),SelectPercentile(input_matrix, SelectPercentile__percentile=20))'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)),
        SelectPercentile(score_func=f_classif, percentile=20)
    ),
    KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_2():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline (only one classifier)."""

    pipeline_string = (
        'KNeighborsClassifier('
        'input_matrix, '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_3():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with a preprocessor."""

    pipeline_string = (
        'DecisionTreeClassifier(SelectPercentile(input_matrix, SelectPercentile__percentile=20),'
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8,'
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)

    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_4():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with input_matrix in CombineDFs."""

    pipeline_string = (
        'KNeighborsClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8,DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),input_matrix)'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1,KNeighborsClassifier__weights=uniform'
    )

    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)),
        FunctionTransformer(copy)
    ),
    KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_5():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with SelectFromModel."""
    pipeline_string = (
        'DecisionTreeRegressor(SelectFromModel(input_matrix, '
        'SelectFromModel__ExtraTreesRegressor__max_features=0.05, SelectFromModel__ExtraTreesRegressor__n_estimators=100, '
        'SelectFromModel__threshold=0.05), DecisionTreeRegressor__max_depth=8,'
        'DecisionTreeRegressor__min_samples_leaf=5, DecisionTreeRegressor__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj_reg._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.05, n_estimators=100), threshold=0.05),
    DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj_reg.operators, tpot_obj_reg._pset)


def test_export_pipeline_6():
    """Assert that exported_pipeline() generated a compile source file with random_state and data_file_path."""

    pipeline_string = (
        'DecisionTreeClassifier(SelectPercentile(input_matrix, SelectPercentile__percentile=20),'
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8,'
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('test_path', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    exported_code = export_pipeline(pipeline, tpot_obj.operators,
                                    tpot_obj._pset, random_state=42,
                                    data_file_path='test_path')

    assert expected_code == exported_code


def test_operator_export():
    """Assert that a TPOT operator can export properly with a callable function as a parameter."""
    assert list(TPOTSelectPercentile.arg_types) == TPOTSelectPercentile_args
    export_string = TPOTSelectPercentile.export(5)
    assert export_string == "SelectPercentile(score_func=f_classif, percentile=5)"


def test_operator_export_2():
    """Assert that a TPOT operator can export properly with a BaseEstimator as a parameter."""
    assert list(TPOTSelectFromModel.arg_types) == TPOTSelectFromModel_args
    export_string = TPOTSelectFromModel.export('gini', 0.10, 100, 0.10)
    expected_string = ("SelectFromModel(estimator=ExtraTreesClassifier(criterion=\"gini\","
        " max_features=0.1, n_estimators=100), threshold=0.1)")
    print(export_string)
    assert export_string == expected_string


def test_get_by_name():
    """Assert that the Operator class returns operators by name appropriately."""

    assert get_by_name("SelectPercentile", tpot_obj.operators).__class__ == TPOTSelectPercentile.__class__
    assert get_by_name("SelectFromModel", tpot_obj.operators).__class__ == TPOTSelectFromModel.__class__


def test_get_by_name_2():
    """Assert that get_by_name raises TypeError with a incorrect operator name."""

    assert_raises(TypeError, get_by_name, "RandomForestRegressor", tpot_obj.operators)
    # use correct name
    ret_op_class = get_by_name("RandomForestClassifier", tpot_obj.operators)


def test_get_by_name_3():
    """Assert that get_by_name raises ValueError with duplicate operators in operator dictionary."""

    # no duplicate
    ret_op_class = get_by_name("SelectPercentile", tpot_obj.operators)
    # add a copy of TPOTSelectPercentile into operator list
    tpot_obj.operators.append(TPOTSelectPercentile)
    assert_raises(ValueError, get_by_name, "SelectPercentile", tpot_obj.operators)


def test_indent():
    """Assert that indenting a multiline string by 4 spaces prepends 4 spaces before each new line."""
    multiline_string = """test
test1
test2
test3"""

    indented_multiline_string = """    test
    test1
    test2
    test3"""

    assert indented_multiline_string == _indent(multiline_string, 4)


def test_pipeline_score_save():
    """Assert that the TPOTClassifier can generate a scored pipeline export correctly."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline_string = (
        'DecisionTreeClassifier(SelectPercentile(input_matrix, SelectPercentile__percentile=20),'
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8,'
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.929813743
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=20),
    DecisionTreeClassifier(criterion="gini", max_depth=8, min_samples_leaf=5, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert_equal(expected_code, export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset, pipeline_score=0.929813743))


def test_imputer_in_export():
    """Assert that TPOT exports a pipeline with an imputation step if imputation was used in fit()."""
    tpot_obj = TPOTClassifier(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0,
        config_dict='TPOT light'
    )
    features_with_nan = np.copy(training_features)
    features_with_nan[0][0] = float('nan')

    tpot_obj.fit(features_with_nan, training_target)
    # use fixed pipeline since the random.seed() performs differently in python 2.* and 3.*
    pipeline_string = (
        'KNeighborsClassifier('
        'input_matrix, '
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')'
    )
    tpot_obj._optimized_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)

    export_code = export_pipeline(tpot_obj._optimized_pipeline, tpot_obj.operators, tpot_obj._pset, tpot_obj._imputed)

    expected_code = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""

    assert_equal(export_code, expected_code)


def test_set_param_recursive():
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    """Assert that _set_param_recursive sets \"random_state\" to 42 in all steps in a simple pipeline."""
    pipeline_string = (
        'DecisionTreeClassifier(PCA(input_matrix, PCA__iterated_power=5, PCA__svd_solver=randomized), '
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )

    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)
    # assert "random_state" of PCA at step 1
    assert getattr(sklearn_pipeline.steps[0][1], 'random_state') == 42
    # assert "random_state" of DecisionTreeClassifier at step 2
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42


def test_set_param_recursive_2():
    """Assert that set_param_recursive sets \"random_state\" to 42 in nested estimator in SelectFromModel."""
    pipeline_string = (
        'DecisionTreeRegressor(SelectFromModel(input_matrix, '
        'SelectFromModel__ExtraTreesRegressor__max_features=0.05, SelectFromModel__ExtraTreesRegressor__n_estimators=100, '
        'SelectFromModel__threshold=0.05), DecisionTreeRegressor__max_depth=8,'
        'DecisionTreeRegressor__min_samples_leaf=5, DecisionTreeRegressor__min_samples_split=5)'
    )
    tpot_obj = TPOTRegressor()
    tpot_obj._fit_init()
    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

    assert getattr(getattr(sklearn_pipeline.steps[0][1], 'estimator'), 'random_state') == 42
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42


def test_set_param_recursive_3():
    """Assert that set_param_recursive sets \"random_state\" to 42 in nested estimator in StackingEstimator in a complex pipeline."""
    pipeline_string = (
        'DecisionTreeClassifier(CombineDFs('
        'DecisionTreeClassifier(input_matrix, DecisionTreeClassifier__criterion=gini, '
        'DecisionTreeClassifier__max_depth=8, DecisionTreeClassifier__min_samples_leaf=5,'
        'DecisionTreeClassifier__min_samples_split=5),input_matrix) '
        'DecisionTreeClassifier__criterion=gini, DecisionTreeClassifier__max_depth=8, '
        'DecisionTreeClassifier__min_samples_leaf=5, DecisionTreeClassifier__min_samples_split=5)'
    )
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()

    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    sklearn_pipeline = tpot_obj._toolbox.compile(expr=deap_pipeline)
    set_param_recursive(sklearn_pipeline.steps, 'random_state', 42)

    # StackingEstimator under the transformer_list of FeatureUnion
    assert getattr(getattr(sklearn_pipeline.steps[0][1].transformer_list[0][1], 'estimator'), 'random_state') == 42
    assert getattr(sklearn_pipeline.steps[1][1], 'random_state') == 42

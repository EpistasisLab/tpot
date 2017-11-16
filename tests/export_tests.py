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
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code, get_by_name
from tpot.operator_utils import TPOTOperatorClassFactory
from tpot.config.classifier import classifier_config_dict

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from deap import creator

from nose.tools import assert_raises, assert_equal

test_operator_key = 'sklearn.feature_selection.SelectPercentile'

TPOTSelectPercentile, TPOTSelectPercentile_args = TPOTOperatorClassFactory(
    test_operator_key,
    classifier_config_dict[test_operator_key]
)

mnist_data = load_digits()
training_features, testing_features, training_target, testing_target = \
    train_test_split(mnist_data.data.astype(np.float64), mnist_data.target.astype(np.float64), random_state=42)


def test_export_random_ind():
    """Assert that the TPOTClassifier can generate the same pipeline export with random seed of 39."""
    tpot_obj = TPOTClassifier(random_state=39)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline = tpot_obj._toolbox.individual()
    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=65),
    DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=4, min_samples_split=18)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    print(export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset))
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export():
    """Assert that TPOT's export function throws a RuntimeError when no optimized pipeline exists."""
    tpot_obj = TPOTClassifier()
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


def test_generate_pipeline_code():
    """Assert that generate_pipeline_code() returns the correct code given a specific pipeline."""
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier(random_state=42)
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
    tpot_obj = TPOTClassifier()
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

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

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
    tpot_obj = TPOTClassifier()
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

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_export_pipeline_3():
    """Assert that exported_pipeline() generated a compile source file as expected given a fixed simple pipeline with a preprocessor."""
    tpot_obj = TPOTClassifier()
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

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

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
    tpot_obj = TPOTClassifier()
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

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

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
    tpot_obj = TPOTRegressor()
    pipeline_string = (
        'DecisionTreeRegressor(SelectFromModel(input_matrix, '
        'SelectFromModel__ExtraTreesRegressor__max_features=0.05, SelectFromModel__ExtraTreesRegressor__n_estimators=100, '
        'SelectFromModel__threshold=0.05), DecisionTreeRegressor__max_depth=8,'
        'DecisionTreeRegressor__min_samples_leaf=5, DecisionTreeRegressor__min_samples_split=5)'
    )
    pipeline = creator.Individual.from_string(pipeline_string, tpot_obj._pset)
    expected_code = """import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

exported_pipeline = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.05, n_estimators=100), threshold=0.05),
    DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
    assert expected_code == export_pipeline(pipeline, tpot_obj.operators, tpot_obj._pset)


def test_operator_export():
    """Assert that a TPOT operator can export properly with a function as a parameter to a classifier."""
    export_string = TPOTSelectPercentile.export(5)
    assert export_string == "SelectPercentile(score_func=f_classif, percentile=5)"


def test_get_by_name():
    """Assert that the Operator class returns operators by name appropriately."""
    tpot_obj = TPOTClassifier()
    assert get_by_name("SelectPercentile", tpot_obj.operators).__class__ == TPOTSelectPercentile.__class__


def test_get_by_name_2():
    """Assert that get_by_name raises TypeError with a incorrect operator name."""
    tpot_obj = TPOTClassifier()
    assert_raises(TypeError, get_by_name, "RandomForestRegressor", tpot_obj.operators)
    # use correct name
    ret_op_class = get_by_name("RandomForestClassifier", tpot_obj.operators)


def test_get_by_name_3():
    """Assert that get_by_name raises ValueError with duplicate operators in operator dictionary."""
    tpot_obj = TPOTClassifier()
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
    tpot_obj = TPOTClassifier(random_state=39)
    tpot_obj._pbar = tqdm(total=1, disable=True)
    pipeline = tpot_obj._toolbox.individual()
    expected_code = """import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.929813743
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=65),
    DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=4, min_samples_split=18)
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
from sklearn.preprocessing import Imputer

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state=42)

imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

exported_pipeline = KNeighborsClassifier(n_neighbors=10, p=1, weights="uniform")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""

    assert_equal(export_code, expected_code)

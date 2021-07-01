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

from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import config_imagefeatureextract
from tpot.builtins import feature_extractors as tpot_fe
from tpot.gp_types import Output_Array, Image_Array

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
try:
    from sklearn.feature_selection._base import SelectorMixin
except ImportError:
    from sklearn.feature_selection.base import SelectorMixin

from nose.tools import nottest, assert_raises
from itertools import repeat

train_test_split = nottest(train_test_split)

# Set up pandas data used in tests
input_data = pd.read_csv(
    'tests/tests.csv',
    sep=',',
    dtype=np.float64,
)
pd_features = input_data.drop('class', axis=1)
pd_target = input_data['class']

multiclass_X, multiclass_y = make_classification(
    n_samples=25,
    n_features=4,
    n_classes=3,
    n_clusters_per_class=1
)

#Set up digits dataset as images (and as flattened version)
digits = load_digits()
X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(digits.images, digits.target, 
                                                    train_size=0.15, test_size=0.85, random_state=42)

X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(digits.data, digits.target,
                                                    train_size=0.15, test_size=0.85, random_state=42)

# Tests

def test_can_generate_individual_with_images1():
    """Assert that TPOTClassifier can generate a pipeline if the input type is passed in as image (CI)"""
    tpot_obj = TPOTClassifier(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())
    assert(len(pipeline1) != 0)

def test_can_generate_individual_with_images2():
    """Assert that TPOTRegressor can generate a pipeline if the input type is passed in as image (CI)"""
    tpot_obj = TPOTRegressor(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())

    assert(len(pipeline1) != 0)

def test_consistent_generate_individual_with_images1():
    """Assert that TPOTClassifier can generate the same pipeline with same random seed if the input type is passed in as image (CI)"""
    tpot_obj = TPOTClassifier(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTClassifier(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline2 = str(tpot_obj._toolbox.individual())
    assert pipeline1 == pipeline2

def test_consistent_generate_individual_with_images2():
    """Assert that TPOTRegressor can generate the same pipeline with same random seed if the input type is passed in as image (CI)"""
    tpot_obj = TPOTRegressor(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline1 = str(tpot_obj._toolbox.individual())
    tpot_obj = TPOTRegressor(input_type="image", random_state=43)
    tpot_obj._fit_init()
    pipeline2 = str(tpot_obj._toolbox.individual())
    assert pipeline1 == pipeline2

def test_construct_template_with_image_input_type():
    """Assert that TPOT can construct a pipeline with a special input type (image) if given a template (CI)"""
    fe_classifier = TPOTClassifier(input_type="image", 
        template='DeepImageFeatureExtractor-Selector-Classifier',
        random_state=42)
    fe_classifier._fit_init()
    pop = fe_classifier._toolbox.population(n=10)
    for deap_pipeline in pop:
        operator_count = fe_classifier._operator_count(deap_pipeline)
        sklearn_pipeline = fe_classifier._toolbox.compile(expr=deap_pipeline)
        assert operator_count == 3
        assert issubclass(sklearn_pipeline.steps[0][1].__class__, tpot_fe.DeepImageFeatureExtractor)
        assert issubclass(sklearn_pipeline.steps[1][1].__class__, SelectorMixin)
        assert issubclass(sklearn_pipeline.steps[2][1].__class__, ClassifierMixin)

def test_generate_pipeline_with_correct_input():
    """Assert that TPOT's gen_grow_safe function will make pipeline with a starting operator that takes the correct input (CI)"""
    tpot_obj = TPOTClassifier(input_type="image", random_state=42)
    tpot_obj._fit_init()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 2)

    assert pipeline[1].args[0] == Image_Array

def test_increase_size_on_complex_input():
    """Assert that TPOT's gen_grow_safe function will make pipeline length at least 2 if no full-length classifiers exist and the input type is complex (image, text, etc.). (CI)"""
    tpot_obj = TPOTClassifier(input_type="image", random_state=99)
    tpot_obj._fit_init()

    pipeline = tpot_obj._gen_grow_safe(tpot_obj._pset, 1, 2)
    pipe_prims = [ind for ind in pipeline if type(ind) is type(pipeline[0])]

    assert len(pipe_prims) > 1
    assert pipe_prims[0].ret == Output_Array


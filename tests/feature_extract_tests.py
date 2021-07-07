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
from tpot.config import config_textfeatureextract
from tpot.builtins import feature_extractors as tpot_fe

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
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


#Set up a text dataset
text_data_file = 'tests/spam_subset_text.csv'
df = pd.read_csv(text_data_file, delimiter=',', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)

X_rawtext = df.v2.values
X_rawtext = X_rawtext.astype(str)
y_text = df.v1
le = LabelEncoder()
y_text = le.fit_transform(y_text)

X_train_text,X_test_text,y_train_text,y_test_text = train_test_split(X_rawtext,y_text,test_size=0.85, 
                                                    random_state=42)


# Tests

#General extractor tests
def test_complain_about_input_type():
    """Assert that TPOT complains if input_type not set but input is text/image (FE)"""
    clf = TPOTClassifier()
    assert_raises(ValueError, clf.fit, X_train_images, y_train_images)


#Image extractor tests
def test_complain_about_image_input():
    """Assert that TPOT complains if image input set but input is actually feature matrix (FE)"""
    clf = TPOTClassifier(input_type="image")
    assert_raises(ValueError, clf.fit, X_train_flat, y_train_flat)

def test_image_extractor_append_config():
    """Assert that passing the image input type appends the image_extractor dict (FE)"""
    clf = TPOTClassifier(input_type="image")
    clf._fit_init()
    assert config_imagefeatureextract.items() <= clf._config_dict.items()

def test_DeepImageFeatureExtractor_output():
    """Assert that deep image feature extractor returns a 2D array with the same number of samples after passing images (FE)"""
    DIFE_test_in = X_train_images[0:10, :, :]
    tpot_DIFE = tpot_fe.DeepImageFeatureExtractor()
    X_extracted = tpot_DIFE.fit_transform(DIFE_test_in)
    assert len(X_extracted.shape) == 2 and X_extracted.shape[0] == DIFE_test_in.shape[0]

def test_FlattenerImageExtractor_output():
    """Assert that flattener extractor returns a 2D array that has been properly flattened (FE)"""
    tpot_FIE = tpot_fe.FlattenerImageExtractor()
    X_flattened = tpot_FIE.fit_transform(X_train_images)
    assert(np.array_equal(X_flattened, X_train_flat))

def test_operator_type_image_extractors():
    """Assert that Image Extractors with special inputs/outputs have attributes indicating so (FE)"""
    assert(hasattr(tpot_fe.FlattenerImageExtractor(), 'expected_input_type') and
        hasattr(tpot_fe.FlattenerImageExtractor(), 'expected_output_type'))

def test_MNIST_performance_images():
    """Assert that TPOT maintains relatively good performance (>=0.80) on the MNIST dataset passed as images using the flattener extractor (FE)"""
    fe_classifier = TPOTClassifier(input_type="image",
        population_size=3,
        offspring_size=3,
        generations=3, 
        template='FlattenerImageExtractor-Classifier',
        max_eval_time_mins=3./60, random_state=42)
    fe_classifier.fit(X_test_images, y_test_images)
    score = fe_classifier.score(X_train_images, y_train_images)
    assert(score >= 0.80)


#Text extractor tests
def test_complain_about_text_input():
    """Assert that TPOT complains if text input set but input is actually feature matrix (FE)"""
    clf = TPOTClassifier(input_type="text")
    assert_raises(ValueError, clf.fit, X_train_flat, y_train_flat)

def test_text_extractor_append_config():
    """Assert that passing the text input type appends the text_extractor dict (FE)"""
    clf = TPOTClassifier(input_type="text")
    clf._fit_init()
    assert config_textfeatureextract.items() <= clf._config_dict.items()

def test_TfidfVectorizerTextExtractor_output():
    """Assert that TfidfVectorizer text extractor returns a 2D array with the same number of samples after passing text (FE)"""
    tpot_tf = tpot_fe.TfidfVectorizerTextExtractor()
    X_extracted = tpot_tf.fit_transform(X_train_text)
    assert len(X_extracted.shape) == 2 and X_extracted.shape[0] == X_train_text.shape[0]

def test_CountVectorizerTextExtractor_output():
    """Assert that CountVectorizer text extractor returns a 2D array with the same number of samples after passing text (FE)"""
    tpot_tf = tpot_fe.CountVectorizerTextExtractor()
    X_extracted = tpot_tf.fit_transform(X_train_text)
    assert len(X_extracted.shape) == 2 and X_extracted.shape[0] == X_train_text.shape[0]

def test_operator_type_text_extractors():
    """Assert that Text Extractors with special inputs/outputs have attributes indicating so (FE)"""
    assert(hasattr(tpot_fe.TfidfVectorizerTextExtractor(), 'expected_input_type') and
        hasattr(tpot_fe.TfidfVectorizerTextExtractor(), 'expected_output_type'))

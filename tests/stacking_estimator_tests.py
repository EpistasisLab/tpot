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

import platform

import numpy as np
from tpot.builtins import StackingEstimator
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from tpot_tests import training_features, training_target, training_features_r, training_target_r
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

def test_StackingEstimator_1():
    """Assert that the StackingEstimator returns transformed X with synthetic features in classification."""
    clf = RandomForestClassifier(random_state=42)
    stack_clf = StackingEstimator(estimator=RandomForestClassifier(random_state=42))
    # fit
    clf.fit(training_features, training_target)
    stack_clf.fit(training_features, training_target)
    # get transformd X
    X_clf_transformed = stack_clf.transform(training_features)

    assert np.allclose(clf.predict(training_features), X_clf_transformed[:, 0])
    assert np.allclose(clf.predict_proba(training_features), X_clf_transformed[:, 1:1 + len(np.unique(training_target))])


def test_StackingEstimator_2():
    """Assert that the StackingEstimator returns transformed X with a synthetic feature in regression."""
    reg = RandomForestRegressor(random_state=42)
    stack_reg = StackingEstimator(estimator=RandomForestRegressor(random_state=42))
    # fit
    reg.fit(training_features_r, training_target_r)
    stack_reg.fit(training_features_r, training_target_r)
    # get transformd X
    X_reg_transformed = stack_reg.transform(training_features_r)

    assert np.allclose(reg.predict(training_features_r), X_reg_transformed[:, 0])


def test_StackingEstimator_3():
    """Assert that the StackingEstimator worked as expected in scikit-learn pipeline in classification."""
    stack_clf = StackingEstimator(estimator=RandomForestClassifier(random_state=42))
    meta_clf = LogisticRegression()
    sklearn_pipeline = make_pipeline(stack_clf, meta_clf)
    # fit in pipeline
    sklearn_pipeline.fit(training_features, training_target)
    # fit step by step
    stack_clf.fit(training_features, training_target)
    X_clf_transformed = stack_clf.transform(training_features)
    meta_clf.fit(X_clf_transformed, training_target)
    # scoring
    score = meta_clf.score(X_clf_transformed, training_target)
    pipeline_score = sklearn_pipeline.score(training_features, training_target)
    assert np.allclose(score, pipeline_score)

    # test cv score
    cv_score = np.mean(cross_val_score(sklearn_pipeline, training_features, training_target, cv=3, scoring='accuracy'))
    known_cv_score = 0.9643652561247217

    assert np.allclose(known_cv_score, cv_score)


def test_StackingEstimator_4():
    """Assert that the StackingEstimator worked as expected in scikit-learn pipeline in regression."""
    stack_reg = StackingEstimator(estimator=RandomForestRegressor(random_state=42))
    meta_reg = Lasso(random_state=42)
    sklearn_pipeline = make_pipeline(stack_reg, meta_reg)
    # fit in pipeline
    sklearn_pipeline.fit(training_features_r, training_target_r)
    # fit step by step
    stack_reg.fit(training_features_r, training_target_r)
    X_reg_transformed = stack_reg.transform(training_features_r)
    meta_reg.fit(X_reg_transformed, training_target_r)
    # scoring
    score = meta_reg.score(X_reg_transformed, training_target_r)
    pipeline_score = sklearn_pipeline.score(training_features_r, training_target_r)
    assert np.allclose(score, pipeline_score)

    # test cv score
    cv_score = np.mean(cross_val_score(sklearn_pipeline, training_features_r, training_target_r, cv=3, scoring='r2'))
    known_cv_score = 0.8216045257587923

    # On some non-amd64 systems such as arm64, a resulting score of
    # 0.8207525232725118 was observed, so we need to add a tolerance there
    if platform.machine() != 'x86_64':
        assert np.allclose(known_cv_score, cv_score, rtol=0.01)
    else:
        assert np.allclose(known_cv_score, cv_score)


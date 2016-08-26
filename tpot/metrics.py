# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

import numpy as np


def balanced_accuracy(estimator, X_test, y_test):
    """Default scoring function: balanced accuracy

    Balanced accuracy computes each class' accuracy on a per-class basis using a
    one-vs-rest encoding, then computes an unweighted average of the class accuracies.

    Parameters
    ----------
    estimator: scikit-learn estimator
        The estimator for which to evaluate the balanced accuracy
    X_test: numpy.ndarray {n_samples, n_features}
        Test data that will be fed to estimator.predict.
    y_test: numpy.ndarray {n_samples, 1}
        Target values for X_test.

    Returns
    -------
    fitness: float
        Returns a float value indicating the `individual`'s balanced accuracy
        0.5 is as good as chance, and 1.0 is perfect predictive accuracy
    """
    y_pred = estimator.predict(X_test)
    all_classes = list(set(np.append(y_test, y_pred)))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = \
            float(sum((y_pred == this_class) & (y_test == this_class))) /\
            float(sum((y_test == this_class)))

        this_class_specificity = \
            float(sum((y_pred != this_class) & (y_test != this_class))) /\
            float(sum((y_test != this_class)))

        this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
        all_class_accuracies.append(this_class_accuracy)

    balanced_accuracy = np.mean(all_class_accuracies)
    return balanced_accuracy

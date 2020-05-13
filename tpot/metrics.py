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

import numpy as np
from sklearn.metrics import make_scorer, SCORERS


def balanced_accuracy(y_true, y_pred):
    """Default scoring function: balanced accuracy.

    Balanced accuracy computes each class' accuracy on a per-class basis using a
    one-vs-rest encoding, then computes an unweighted average of the class accuracies.

    Parameters
    ----------
    y_true: numpy.ndarray {n_samples}
        True class labels
    y_pred: numpy.ndarray {n_samples}
        Predicted class labels by the estimator

    Returns
    -------
    fitness: float
        Returns a float value indicating the individual's balanced accuracy
        0.5 is as good as chance, and 1.0 is perfect predictive accuracy
    """
    all_classes = np.unique(np.append(y_true, y_pred))
    all_class_accuracies = []
    for this_class in all_classes:
        this_class_sensitivity = 0.
        this_class_specificity = 0.

        if sum(y_true == this_class) != 0:
            this_class_sensitivity = \
                float(sum((y_pred == this_class) & (y_true == this_class))) /\
                float(sum((y_true == this_class)))
        if sum(y_true != this_class) != 0:
            this_class_specificity = \
                float(sum((y_pred != this_class) & (y_true != this_class))) /\
                float(sum((y_true != this_class)))
        else: # in rase case, y_true has only 1 class then specificity should be 1
            this_class_specificity = 1.

        this_class_accuracy = (this_class_sensitivity + this_class_specificity) / 2.
        all_class_accuracies.append(this_class_accuracy)

    return np.mean(all_class_accuracies)


SCORERS['balanced_accuracy'] = make_scorer(balanced_accuracy)

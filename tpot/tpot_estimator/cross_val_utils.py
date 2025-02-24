"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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
import time
import sklearn.metrics
from collections.abc import Iterable
import pandas as pd
import sklearn
import numpy as np

def cross_val_score_objective(estimator, X, y, scorers, cv, fold=None):
    """
    Compute the cross validated scores for a estimator. Only fits the estimator once per fold, and loops over the scorers to evaluate the estimator.

    Parameters
    ----------
    estimator: sklearn.base.BaseEstimator
        The estimator to fit and score.
    X: np.ndarray or pd.DataFrame
        The feature matrix.
    y: np.ndarray or pd.Series
        The target vector.
    scorers: list or scorer
        The scorers to use. 
        If a list, will loop over the scorers and return a list of scorers.
        If a single scorer, will return a single score.
    cv: sklearn cross-validator
        The cross-validator to use. For example, sklearn.model_selection.KFold or sklearn.model_selection.StratifiedKFold.
    fold: int, optional
        The fold to return the scores for. If None, will return the mean of all the scores (per scorer). Default is None.
    
    Returns
    -------
    scores: np.ndarray or float
        The scores for the estimator per scorer. If fold is None, will return the mean of all the scores (per scorer).
        Returns a list if multiple scorers are used, otherwise returns a float for the single scorer.

    """
    
    #check if scores is not iterable
    if not isinstance(scorers, Iterable): 
        scorers = [scorers]
    scores = []
    if fold is None:
        for train_index, test_index in cv.split(X, y):
            this_fold_estimator = sklearn.base.clone(estimator)
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]

            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                y_train, y_test = y[train_index], y[test_index]


            start = time.time()
            this_fold_estimator.fit(X_train,y_train)
            duration = time.time() - start

            this_fold_scores = [sklearn.metrics.get_scorer(scorer)(this_fold_estimator, X_test, y_test) for scorer in scorers] 
            scores.append(this_fold_scores)
            del this_fold_estimator
            del X_train
            del X_test
            del y_train
            del y_test
            

        return np.mean(scores,0)
    else:
        this_fold_estimator = sklearn.base.clone(estimator)
        train_index, test_index = list(cv.split(X, y))[fold]
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]

        start = time.time()
        this_fold_estimator.fit(X_train,y_train)
        duration = time.time() - start
        this_fold_scores = [sklearn.metrics.get_scorer(scorer)(this_fold_estimator, X_test, y_test) for scorer in scorers] 
        return this_fold_scores










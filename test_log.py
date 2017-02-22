from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
from tqdm import tqdm

from functools import partial
from pathos.multiprocessing import ProcessPool

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.95, test_size=0.05)

pbar = tqdm(total=10, unit='pipeline', leave=False,
                  disable=False, desc='Optimization Progress')

pipeline = make_pipeline(StandardScaler(), GaussianNB(priors=None))

sklearn_pipelines = [pipeline]*10

def _wrapped_cross_val_score(sklearn_pipeline, features, classes,
cv, pbar):
import numpy as np
    cv_scores = cross_val_score(sklearn_pipeline, features, classes,
        cv=cv, n_jobs=1)
    return np.mean(cv_scores)


partial_cross_val_score = partial(_wrapped_cross_val_score, features=X_train, classes=y_train, cv=3, pbar = pbar)
pool = ProcessPool(processes=2)
resulting_score_list = pool.imap(partial_cross_val_score, sklearn_pipelines)
pbar = tqdm(total=10, unit='pipeline', leave=False,
                  disable=False, desc='Optimization Progress')
while True:
    pbar.update(resulting_score_list._index - pbar.n)
    if resulting_score_list._index < 10:
        pass
        print(len(resulting_score_list._items))
    else:
        break

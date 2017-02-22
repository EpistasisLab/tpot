from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
from tqdm import tqdm


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,train_size=0.25, test_size=0.75)
global pbar

pbar = tqdm(total=10, unit='pipeline', leave=False,
                  disable=False, desc='Optimization Progress')

pipeline = make_pipeline(StandardScaler(), GaussianNB(priors=None))

sklearn_pipelines = [pipeline]*10

def _wrapped_cross_val_score(sklearn_pipeline, features, classes,
cv, pbar):
    cv_scores = cross_val_score(sklearn_pipeline, features, classes,
        cv=cv, n_jobs=1)
    return cv_scores

parallel = Parallel(n_jobs=2)
resulting_score_list = parallel(delayed(_wrapped_cross_val_score)(clone(sklearn_pipeline),
            features=X_train, classes=y_train, cv=3, pbar = pbar)
            for sklearn_pipeline in sklearn_pipelines)

from functools import partial
from pathos.multiprocessing import ProcessPool
partial_cross_val_score = partial(_wrapped_cross_val_score, features=X_train, classes=y_train, cv=3, pbar = pbar)
pool = ProcessPool(processes=2)
resulting_score_list = pool.imap(partial_cross_val_score, sklearn_pipelines)
num_done  = 0
pbar = tqdm(total=10, unit='pipeline', leave=False,
                  disable=False, desc='Optimization Progress')
while True:

    if resulting_score_list._index < 10:
        pass
        #print(resulting_score_list._index)
    else:
        break

    num_update = resulting_score_list._index - num_done
    pbar.update(resulting_score_list._index - pbar.n)
    num_done = resulting_score_list._index

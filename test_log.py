# coding: utf-8
get_ipython().magic('load tpot_test_multi_process.py')
# %load tpot_test_multi_process.py
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.25, test_size=0.75)

#tpot = TPOTClassifier(generations=3, population_size=10, verbosity=2, num_cpu=1, random_state = 42)
#time_start = time.time()
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#print('\nTime used with num_cpu = 1:',time.time()-time_start)

tpot = TPOTClassifier(generations=2, population_size=10, verbosity=2, max_eval_time_mins=0.02, n_jobs = 2, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
print('\nTime used with num_cpu = 3:',time.time()-time_start)
tpot.sklearn_pipeline_list[0]
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, clone
from sklearn.externals.joblib import Parallel, delayed
parallel = Parallel(n_jobs=self.n_jobs)
parallel = Parallel(n_jobs=2)
resulting_score_list = parallel(delayed(cross_val_score)(clone(sklearn_pipeline),
            features=X, classes=y, cv=3, njobs = 1)
            for sklearn_pipeline in sklearn_pipeline_list)
resulting_score_list = parallel(delayed(cross_val_score)(clone(sklearn_pipeline),
            features=X, classes=y, cv=3, njobs = 1)
            for sklearn_pipeline in tpot.sklearn_pipeline_list)
resulting_score_list = parallel(delayed(cross_val_score)(clone(sklearn_pipeline),
            features=X_train, classes=y_train, cv=3, njobs = 1)
            for sklearn_pipeline in tpot.sklearn_pipeline_list)
resulting_score_list = parallel(delayed(cross_val_score)(clone(sklearn_pipeline),
            X=X_train, y=y_train, cv=3, njobs = 1)
            for sklearn_pipeline in tpot.sklearn_pipeline_list)
resulting_score_list = parallel(delayed(cross_val_score)(clone(sklearn_pipeline),
            X=X_train, y=y_train, cv=3, n_jobs = 1)
            for sklearn_pipeline in tpot.sklearn_pipeline_list)
partial_cross_val_score = partial(cross_val_score, X=X_train, y=y_train,
            cv=5, n_jobs=1)
from functools import partial
from pathos.multiprocessing import Pool
partial_cross_val_score = partial(cross_val_score, X=X_train, y=y_train,
            cv=5, n_jobs=1)
pool = Pool(processes=2)
resulting_score_list = pool.map(partial_cross_val_score, tpot.sklearn_pipeline_list)
resulting_score_list

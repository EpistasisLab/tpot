from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
import numpy as np
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.25, test_size=0.75)

#tpot = TPOTClassifier(generations=3, population_size=10, verbosity=2, num_cpu=1, random_state = 42)
#time_start = time.time()
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#print('\nTime used with num_cpu = 1:',time.time()-time_start)

training_classes_r_weight = np.array(range(1, len(y_train)+1))

tpot = TPOTClassifier(generations=5, population_size=5, offspring_size=5, verbosity=2, max_eval_time_mins=0.06, n_jobs = 2, random_state = 44)#, max_time_mins=1)
time_start = time.time()
tpot.fit(X_train, y_train, sample_weight=training_classes_r_weight)
print(tpot.score(X_test, y_test))
print('\nTime used with num_cpu = 3:',time.time()-time_start)

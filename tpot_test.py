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

tpot = TPOTClassifier(generations=2, population_size=5, verbosity=3, n_jobs = 3, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
print('\nTime used with num_cpu = 3:',time.time()-time_start)

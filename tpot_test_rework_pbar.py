from tpot import TPOTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=3, population_size=10, lamda=20, verbosity=1, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
print('\nTime used',time.time()-time_start)

tpot = TPOTClassifier(generations=3, population_size=10, lamda=20, verbosity=2, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
print('\nTime used',time.time()-time_start)

tpot = TPOTClassifier(generations=3, population_size=10, lamda=20, verbosity=3, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
print('\nTime used',time.time()-time_start)

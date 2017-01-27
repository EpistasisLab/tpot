from tpot import TPOTClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

X, y = make_classification(n_samples=200, n_features=50,
                                            n_informative=10, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)


tpot = TPOTClassifier(generations=4, population_size=20, verbosity=2, random_state = 42)
time_start = time.time()
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_export_class_conf.py')
print('\nTime usages:',time.time()-time_start)

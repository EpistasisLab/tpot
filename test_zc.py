from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOT(generations=2, population_size=10, verbosity=2)
tpot.fit(X_train, y_train)
print('Score: {}'.format(tpot.score(X_test, y_test)))
tpot.export('tpot_mnist_pipeline.py')

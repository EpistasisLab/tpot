from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.10, test_size=0.90)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

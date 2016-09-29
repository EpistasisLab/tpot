from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                            train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=3, population_size=40, verbosity=2,max_eval_time_mins=0.02,random_state=99)
tpot.fit(X_train, y_train)




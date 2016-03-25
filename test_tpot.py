from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
import pandas as pd

#digits = load_digits(n_class=10)
digits = pd.read_csv('./data/mnist.csv', header=0, index_col=False, sep='\t').sample(n=5000, random_state=42)

print(digits.columns)
class_col = digits.columns[0]
data_cols = digits.columns[1:]
X_train, X_test, y_train, y_test = train_test_split(digits[data_cols].values, digits[class_col].values,
                                                            train_size=0.80)

for x in list(range(10)):
    tpot = TPOT(generations=5, verbosity=2, population_size=100)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

    tpot.export('exported_code.txt')


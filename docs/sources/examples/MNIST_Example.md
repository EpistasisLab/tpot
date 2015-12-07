# MNIST Example

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train, X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves ~97% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```python
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75)))


result1 = tpot_data.copy()

# Perform classification with a logistic regression classifier
lrc1 = LogisticRegression(C=2.8214285714285716)
lrc1.fit(result1.loc[training_indeces].drop('class', axis=1).values, result1.loc[training_indeces, 'class'].values)
result1['lrc1-classification'] = lrc1.predict(result1.drop('class', axis=1).values)
```

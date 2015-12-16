# MNIST Example

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size = 0.25)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train, X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves ~97% testing accuracy. Please note that sometimes when both train_size and test_size aren't specified in train_test_split() calls, the split doesn't use the entire data set. So we need to specify both.  

For details on how the `fit()`, `score()` and `export()` functions work, please see [here](Using_TPOT_via_code.md)

After running the above code, the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```python
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size = 0.25)))


result1 = tpot_data.copy()

# Perform classification with a logistic regression classifier
lrc1 = LogisticRegression(C=0.1)
lrc1.fit(result1.loc[training_indeces].drop('class', axis=1).values, result1.loc[training_indeces, 'class'].values)
result1['lrc1-classification'] = lrc1.predict(result1.drop('class', axis=1).values)

# Decision-tree based feature selection
training_features = result1.loc[training_indeces].drop('class', axis=1)
training_class_vals = result1.loc[training_indeces, 'class'].values

pair_scores = dict()
for features in combinations(training_features.columns.values, 2):
    dtc = DecisionTreeClassifier()
    training_feature_vals = training_features[list(features)].values
    dtc.fit(training_feature_vals, training_class_vals)
    pair_scores[features] = (dtc.score(training_feature_vals, training_class_vals), list(features))

best_pairs = []
for pair in sorted(pair_scores, key=pair_scores.get, reverse=True)[:3870]:
    best_pairs.extend(list(pair))
best_pairs = sorted(list(set(best_pairs)))

result2 = result1[sorted(list(set(best_pairs + ['class'])))]

# Perform classification with a random forest classifier
rfc3 = RandomForestClassifier(n_estimators=1, max_features=min(64, len(result2.columns) - 1))
rfc3.fit(result2.loc[training_indeces].drop('class', axis=1).values, result2.loc[training_indeces, 'class'].values)
result3 = result2
result3['rfc3-classification'] = rfc3.predict(result3.drop('class', axis=1).values)

# Perform classification with a decision tree classifier
dtc4 = DecisionTreeClassifier(max_features=min(40, len(result3.columns) - 1), max_depth=7)
dtc4.fit(result3.loc[training_indeces].drop('class', axis=1).values, result3.loc[training_indeces, 'class'].values)
result4 = result3
result4['dtc4-classification'] = dtc4.predict(result4.drop('class', axis=1).values)

# Perform classification with a k-nearest neighbor classifier
knnc5 = KNeighborsClassifier(n_neighbors=1)
knnc5.fit(result4.loc[training_indeces].drop('class', axis=1).values, result4.loc[training_indeces, 'class'].values)
result5 = result4
result5['knnc5-classification'] = knnc5.predict(result5.drop('class', axis=1).values)
```
# Count of Zero elements
* * *

Counts number of elements per row that are equal to zero, and adds the count as
a feature. Also adds a feature for the count of non-zero elements.

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_constructed_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing new 'non_zero' and 'zero_col' features.

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('iris.csv', sep=',')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a k-nearest neighbor classifier
knnc1 = KNeighborsClassifier(n_neighbors=min(10, len(training_indices)))
knnc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['knnc1-classification'] = knnc1.predict(result1.drop('class', axis=1).values)

# Add virtual features for number of zeros and non-zeros per row
feature_cols_only = result1.loc[training_indices].drop('class', axis=1)

if len(feature_cols_only.columns.values) > 0:
    non_zero_col = np.array([np.count_nonzero(row) for i, row in result1.iterrows()]).astype(np.float64)
    zero_col     = np.array([(len(feature_cols_only.columns.values) - x) for x in non_zero_col]).astype(np.float64)

    result2 = result1.copy()
    result2['non_zero'] = pd.Series(non_zero_col, index=result2.index)
    result2['zero_col'] = pd.Series(zero_col, index=result2.index)
    result2['class'] = result1['class'].values
else:
    result2 = result1.copy()

# Perform classification with a decision tree classifier
dtc3 = DecisionTreeClassifier(max_features=min(1198, len(result2.columns) - 1), max_depth=93)
dtc3.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)
result3 = result2.copy()
result3['dtc3-classification'] = dtc3.predict(result3.drop('class', axis=1).values)
```

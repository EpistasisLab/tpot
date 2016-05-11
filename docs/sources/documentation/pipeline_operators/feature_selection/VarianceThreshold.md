# Variance Threshold
* * *

Uses Scikit-learn's VarianceThreshold feature selection to learn the subset of features that pass the variance threshold.

## Dependencies
    sklearn.feature_selection.VarianceThreshold

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    threshold: float
        The variance threshold that removes features that fall under the threshold

Returns
-------
    subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the features that are above the variance threshold

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


# Use Scikit-learn's VarianceThreshold for feature selection
training_features = tpot_data.loc[training_indices].drop('class', axis=1)

selector = VarianceThreshold(threshold=0.50)
try:
    selector.fit(training_features.values)
except ValueError:
    # None of the features meet the variance threshold
    result1 = tpot_data[['class']]

mask = selector.get_support(True)
mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
result1 = tpot_data[mask_cols]

# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

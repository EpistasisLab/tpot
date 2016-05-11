# Select K-Best
* * *

Uses Scikit-learn's SelectKBest feature selection to learn the subset of features that have the highest score according to some scoring function.

## Dependencies
    sklearn.feature_selection.SelectKBest
    sklearn.feature_selection.f_classif


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    k: int
        The top k features to keep from the original set of features in the training data

Returns
-------
    subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the `k` best features

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)

# Use Scikit-learn's SelectKBest for feature selection
training_features = tpot_data.loc[training_indices].drop('class', axis=1)
training_class_vals = tpot_data.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    result1 = tpot_data.copy()
else:
    selector = SelectKBest(f_classif, k=1)
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    result1 = tpot_data[mask_cols]


# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

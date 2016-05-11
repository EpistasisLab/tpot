# Recursive Feature Elimination
* * * 

Uses Scikit-learn's Recursive Feature Elimintation to learn the subset of features that have the highest weights according to the estimator.

## Dependencies 
    sklearn.feature_selection.RFE
    sklearn.svm.SVC


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    num_features: int
        The number of features to select
    step: float
        The percentage of features to drop each iteration

Returns
-------
    subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the `num_features` best features

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)



# Use Scikit-learn's Recursive Feature Elimination (RFE) for feature selection
training_features = tpot_data.loc[training_indices].drop('class', axis=1)
training_class_vals = tpot_data.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    result1 = tpot_data.copy()
else:
    selector = RFE(SVC(kernel='linear'), n_features_to_select=1, step=0.99)
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

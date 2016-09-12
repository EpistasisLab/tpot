# Standard Feature Scaler
* * *

Uses Scikit-learn's StandardScaler to scale the features by removing their mean and scaling to unit variance.

## Dependencies 
    sklearn.preprocessing.StandardScaler

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale

Returns
-------
    scaled_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the scaled features

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


# Use Scikit-learn's StandardScaler to scale the features
training_features = tpot_data.loc[training_indices].drop('class', axis=1)
result1 = tpot_data.copy()

if len(training_features.columns.values) > 0:
    scaler = StandardScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result1 = pd.DataFrame(data=scaled_features)
    result1['class'] = tpot_data['class'].values


# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

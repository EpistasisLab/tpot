# Polynomial Features
* * *

Uses Scikit-learn's PolynomialFeatures to construct new degree-2 polynomial features from the existing feature set.

## Dependencies 
    sklearn.preprocessing.PolynomialFeatures

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_constructed_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the constructed features

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


# Use Scikit-learn's PolynomialFeatures to construct new features from the existing feature set
training_features = tpot_data.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0 and len(training_features.columns.values) <= 700:
    # The feature constructor must be fit on only the training data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(training_features.values.astype(np.float64))
    constructed_features = poly.transform(tpot_data.drop('class', axis=1).values.astype(np.float64))

    tpot_data_classes = tpot_data['class'].values
    result1 = pd.DataFrame(data=constructed_features)
    result1['class'] = tpot_data_classes
else:
    result1 = tpot_data.copy()

# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

# Radial Basis Function Kernel Sampler
* * *

Uses Scikit-learn's RBFSampler to transform the feature set.

## Dependencies
    sklearn.kernel_approximation.RBFSampler


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    gamma:
      Parameter of RBF kernel: exp(-gamma * x^2)

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features


Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

# Use Scikit-learn's RBFSampler to transform the feature set
training_features = result1.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # RBF must be fit on only the training data
    rbf = RBFSampler(gamma=0.001)
    rbf.fit(training_features.values.astype(np.float64))
    transformed_features = rbf.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result1 = pd.DataFrame(data=transformed_features)
    result1['class'] = tpot_data['class'].values
else:
    result1 = tpot_data.copy()

# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

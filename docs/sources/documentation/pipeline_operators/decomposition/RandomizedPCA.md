# Randomized Principal Component Analysis
* * *

Uses Scikit-learn's RandomizedPCA to transform the feature set.

## Dependencies 
    sklearn.decomposition.RandomizedPCA


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    n_components: int
        The number of components to keep
    iterated_power: int
        Number of iterations for the power method. [1, 10]

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features


Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import RandomizedPCA

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))


# Use Scikit-learn's RandomizedPCA to transform the feature set
training_features = tpot_data.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # RandomizedPCA must be fit on only the training data
    pca = RandomizedPCA(n_components=1, iterated_power=10)
    pca.fit(training_features.values.astype(np.float64))
    transformed_features = pca.transform(tpot_data.drop('class', axis=1).values.astype(np.float64))

    tpot_data_classes = tpot_data['class'].values
    result1 = pd.DataFrame(data=transformed_features)
    result1['class'] = tpot_data_classes
else:
    result1 = tpot_data.copy()

# Perform classification with a decision tree classifier
result2 = result1.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)

result2['dtc1-classification'] = dtc1.predict(result2.drop('class', axis=1).values)

```

# k-Nearest Neighbors Classifier

## Dependencies
    sklearn.neighbors.KNeighborsClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the k-nearest neighbor classifier
    n_neighbors: int
        Number of neighbors to use by default for k_neighbors queries; must be a positive value

Returns
-------
    input_df: pandas.DataFrame {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
        Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
        Also adds the classifiers's predictions as a 'SyntheticFeature' column.

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))

result1 = tpot_data.copy()

# Perform classification with a k-nearest neighbor classifier
knnc1 = KNeighborsClassifier(n_neighbors=2)
knnc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['knnc1-classification'] = knnc1.predict(result1.drop('class', axis=1).values)

```

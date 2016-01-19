# Principal Component Analysis
<<<<<<< HEAD
* * *

Uses Scikit-learn's PCA to transform the feature set.
=======
>>>>>>> 349383d0e1000a92218470a6a3a62e13704d8431

## Dependencies 
    sklearn.decomposition.PCA


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    n_components: int
        The number of components to keep

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features


Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))


# Use Scikit-learn's PCA to transform the feature set
training_features = tpot_data.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # PCA must be fit on only the training data
    pca = PCA(n_components=1)
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

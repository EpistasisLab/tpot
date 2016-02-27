# XGBoost Classifier
* * *

Fits the dmlc eXtreme gradient boosting classifier.

## Dependencies
    xgboost.XGBClassifier


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the XGBoost classifier
    learning_rate: float
        Shrinks the contribution of each tree by learning_rate
    n_estimators: int
        The number of boosting stages to perform
    max_depth: int
        Maximum depth of the individual estimators; the maximum depth limits the number of nodes in the tree

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
from sklearn.cross_validation import StratifiedShuffleSplit
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))

result1 = tpot_data.copy()

# Perform classification with a gradient boosting classifier
xgbc1 = XGBClassifier(learning_rate=0.0001, n_estimators=1, max_depth=None)
xgbc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['xgbc1-classification'] = xgbc1.predict(result1.drop('class', axis=1).values)

```

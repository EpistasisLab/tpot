# Gradient Boosting Classifier
* * *

Fits a Gradient Boosting classifier.

## Dependencies
     sklearn.ensemble.GradientBoostingClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the random forest
    learning_rate: float
        Learning rate shrinks the contribution of each tree by learning_rate
    max_depth: int
        Maximum depth of the individual regression estimators

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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a gradient boosting classifier
gbc1 = GradientBoostingClassifier(learning_rate=1.0, max_depth=3, n_estimators=500, random_state=42)
gbc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['gbc1-classification'] = gbc1.predict(result1.drop('class', axis=1).values)

```

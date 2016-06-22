# AdaBoost Classifier
* * *

Fits an AdaBoost classifier.

## Dependencies
    sklearn.ensemble.AdaBoostClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    learning_rate: float
        Learning rate shrinks the contribution of each classifier by learning_rate.

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
from sklearn.ensemble import AdaBoostClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

adab1 = AdaBoostClassifier(learning_rate=0.1, n_estimators=500, random_state=42)
adab1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['adab1-classification'] = adab1.predict(result1.drop('class', axis=1).values)

```

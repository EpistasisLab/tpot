# Logistic Regression
* * * 

Fits a Logistic Regression classifier

## Dependencies
    sklearn.linear_model.LogisticRegression

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the logistic regression classifier
    C: float
        Inverse of regularization strength; must be a positive value. Like in support vector machines, smaller values specify stronger regularization.

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
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))

result1 = tpot_data.copy()

# Perform classification with a logistic regression classifier
lrc1 = LogisticRegression(C=0.0001)
lrc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['lrc1-classification'] = lrc1.predict(result1.drop('class', axis=1).values)
```

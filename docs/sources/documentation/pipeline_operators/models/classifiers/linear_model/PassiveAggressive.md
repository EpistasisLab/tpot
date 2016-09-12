# Passive Aggressive Classifier
* * *

Fits a Passive Aggressive classifier

## Dependencies
    sklearn.linear_model.PassiveAggressiveClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    C: float
        Penalty parameter C of the error term.
    loss: int
        Integer used to determine the loss function (either 'hinge' or 'squared_hinge')

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
from sklearn.linear_model import PassiveAggressiveClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

pagr1 = PassiveAggressiveClassifier(C=1.0, loss='hinge', fit_intercept=True, random_state=42)
pagr1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['etc1-classification'] = pagr1.predict(result1.drop('class', axis=1).values)

```

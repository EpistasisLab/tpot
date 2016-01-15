# Random Forest Classifier
* * * 

## Dependencies 
     sklearn.ensemble.RandomForestClassifier


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the random forest
    n_estimators: int
        Number of trees in the random forest; must be a positive value
    max_features: int
        Number of features used to fit the decision tree; must be a positive value

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

# Perform classification with a random forest classifier
rfc1 = RandomForestClassifier(n_estimators=1, max_features='auto')
rfc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['rfc1-classification'] = rfc1.predict(result1.drop('class', axis=1).values)

```

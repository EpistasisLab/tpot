# C-Support Vector Classifier

## Dependencies
    sklearn.svm.SVC

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the C-support vector classifier
    C: float
        Penalty parameter C of the error term; must be a positive value

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

# Perform classification with a C-support vector classifier
svc1 = SVC(C=0.0001)
svc1.fit(df.loc[training_indices].drop('class', axis=1).values, df.loc[training_indices, 'class'].values)

res = df
res['svc1-classification'] = svc1.predict(res.drop('class', axis=1).values)

```

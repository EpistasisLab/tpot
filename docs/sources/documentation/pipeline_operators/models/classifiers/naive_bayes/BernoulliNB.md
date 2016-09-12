# BernoulliNB Classifier
* * *

Fits a Naive Bayes classifier for multivariate Bernoulli models.

## Dependencies
    sklearn.naive_bayes.BernoulliNB

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    alpha: float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    binarize: float
        Threshold for binarizing (mapping to booleans) of sample features.

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
from sklearn.naive_bayes import BernoulliNB

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

bnb1 = BernoulliNB(alpha=0.01, binarize=1.0, fit_prior=True)
bnb1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['bnb1-classification'] = bnb1.predict(result1.drop('class', axis=1).values)

```

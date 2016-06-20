# Extra Trees Classifier
* * *

Fits an extra-trees classifier.

## Dependencies
    sklearn.ensemble.ExtraTreesClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    criterion: int
        Integer that is used to select from the list of valid criteria,
        either 'gini', or 'entropy'
    max_features: float
        The number of features to consider when looking for the best split
    min_weight_fraction_leaf: float
        The minimum weighted fraction of the input samples required to be at a leaf node.

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
from sklearn.ensemble import ExtraTreesClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

etc1 = ExtraTreesClassifier(n_estimators=500, random_state=42, max_features=1., min_weight_fraction_leaf=0.1, criterion='gini')
etc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['etc1-classification'] = etc1.predict(result1.drop('class', axis=1).values)

```

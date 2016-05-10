# Decision Tree Classifier
* * * 

Fits a Decision Tree classifier

## Dependencies 
    sklearn.tree.DecisionTreeClassifier

Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the decision tree
    max_features: int
        Number of features used to fit the decision tree; must be a positive value
    max_depth: int
        Maximum depth of the decision tree; must be a positive value

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
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)
    modified_df[non_feature_column] = input_df[non_feature_column].values

# Perform classification with a decision tree classifier
result1 = tpot_data.copy()

dtc1 = DecisionTreeClassifier(max_features='auto', max_depth=None)
dtc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['dtc1-classification'] = dtc1.predict(result1.drop('class', axis=1).values)

```

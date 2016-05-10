# Agglomerate Features
* * *

Uses Scikit-learn's FeatureAgglomeration to transform the feature set.

## Dependencies
    sklearn.cluster.FeatureAgglomeration


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    n_clusters: int
        The number of clusters to find.
    affinity: int
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted.
        Input integer is used to select one of the above strings.
    linkage: int
        Can be one of the following values:
            "ward", "complete", "average"
        Input integer is used to select one of the above strings.

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features

Example Exported Code
---------------------

```Python
import numpy as np
import pandas as pd

from sklearn.cluster import FeatureAgglomeration
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Use Scikit-learn's FeatureAgglomeration to transform the feature set
training_features = result1.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # FeatureAgglomeration must be fit on only the training data
    fa = FeatureAgglomeration(n_clusters=51, affinity='euclidean', linkage='complete')
    fa.fit(training_features.values.astype(np.float64))
    transformed_features = fa.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result1 = pd.DataFrame(data=transformed_features)
    result1['class'] = result1['class'].values
else:
    result1 = result1.copy()

# Perform classification with a decision tree classifier
dtc2 = DecisionTreeClassifier(max_features=min(145, len(result1.columns) - 1), max_depth=2835)
dtc2.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result2 = result1.copy()
result2['dtc2-classification'] = dtc2.predict(result2.drop('class', axis=1).values)

```

# Recursive Feature Elimination
* * *

Uses Scikit-learn's Recursive Feature Elimintation to learn the subset of features that have the highest weights according to the estimator.

## Dependencies
    sklearn.feature_selection.RFE
    sklearn.svm.SVC


Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    num_features: int
        The number of features to select
    step: float
        The percentage of features to drop each iteration

Returns
-------
    subsetted_df: numpy.ndarray {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the `num_features` best features

Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)


exported_pipeline = make_pipeline(
    RFE(estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=42, shrinking=True,
      tol=0.001, verbose=False), step=0.96),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)

```

# Select Percentile
* * *

Uses Scikit-learn's SelectPercentile feature selection to learn the subset of features that belong in the highest `percentile` according to a given scoring function..

## Dependencies
    sklearn.feature_selection.SelectPercentile
    sklearn.feature_selection.f_classif


Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    percentile: int
        The features that belong in the top percentile to keep from the original set of features in the training data

Returns
-------
    subsetted_df: numpy.ndarray {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the best features in the given `percentile`

Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1), input_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectPercentile(percentile=38, score_func=f_classif),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)

```

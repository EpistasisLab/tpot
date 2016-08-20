# Select Fwe
* * *

Uses Scikit-learn's SelectFWE feature selection to learn the subset of features that have the highest score according to some scoring function.

## Dependencies
    sklearn.feature_selection.SelectFwe
    sklearn.feature_selection.f_classif


Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to perform feature selection on
    alpha: float in the range [0.001, 0.05]
        The highest uncorrected p-value for features to keep

Returns
-------
    subsetted_df: numpy.ndarray {n_samples, n_filtered_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the `k` best features

Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1), input_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    SelectFwe(alpha=0.05, score_func=f_classif),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

# Gradient Boosting Classifier
* * *

Fits a Gradient Boosting classifier.

## Dependencies
     sklearn.ensemble.GradientBoostingClassifier

Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the XGBoost classifier
    learning_rate: float
        Shrinks the contribution of each tree by learning_rate
    max_features: float
        Maximum number of features to use (proportion of total features)
    min_weight_fraction_leaf: float
        The minimum weighted fraction of the input samples required to be at a leaf node.

Returns
-------
    input_df: numpy.ndarray {n_samples, n_features+['guess', 'group', 'class', 'SyntheticFeature']}
        Returns a modified input DataFrame with the guess column updated according to the classifier's predictions.
        Also adds the classifiers's predictions as a 'SyntheticFeature' column.

Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1), input_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)


exported_pipeline = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.96, max_features=0.96, min_weight_fraction_leaf=0.08, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

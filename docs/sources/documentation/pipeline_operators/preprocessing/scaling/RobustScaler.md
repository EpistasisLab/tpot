# Robust Feature Scaler
* * *

Uses Scikit-learn's RobustScaler to scale the features using statistics that are robust to outliers.

## Dependencies
    sklearn.preprocessing.RobustScaler

Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale

Returns
-------
    scaled_df: numpy.ndarray {n_samples, n_features + ['guess', 'group', 'class']}
        Returns a DataFrame containing the scaled features

Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)


exported_pipeline = make_pipeline(
    RobustScaler(copy=False),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

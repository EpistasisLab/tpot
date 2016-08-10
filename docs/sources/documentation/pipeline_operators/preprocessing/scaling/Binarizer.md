# Binarizer
* * *

Uses Scikit-learn's Binarizer to binarize all of the features, setting any feature >`threshold` to 1 and all others to 0

## Dependencies
    sklearn.preprocessing.Binarizer

Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    threshold: float
            Feature values below or equal to this value are replaced by 0, above it by 1

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
from sklearn.preprocessing import Binarizer, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')

features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    Binarizer(copy=False, threshold=0.96),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

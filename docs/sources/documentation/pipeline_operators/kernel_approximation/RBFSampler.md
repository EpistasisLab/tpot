# Radial Basis Function Kernel Sampler
* * *

Uses Scikit-learn's RBFSampler to transform the feature set.

## Dependencies
    sklearn.kernel_approximation.RBFSampler


Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    gamma:
      Parameter of RBF kernel: exp(-gamma * x^2)

Returns
-------
    modified_df: numpy.ndarray {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features


Example Exported Code
---------------------

```Python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')

features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.96),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

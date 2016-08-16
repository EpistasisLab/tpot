# LinearSVC Classifier
* * *

Fits a Linear Support Vector Classifier

## Dependencies
    sklearn.svm.LinearSVC

Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    C: float
        Penalty parameter C of the error term.
    penalty: int
        Integer used to specify the norm used in the penalization (l1 or l2)
    dual: bool
        Select the algorithm to either solve the dual or primal optimization problem.

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
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC


# NOTE: Make sure that the class is labeled 'class' in the data file
input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1), input_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    LinearSVC(C=0.96, dual=False, penalty="l1")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

# BernoulliNB Classifier
* * *

Fits a Naive Bayes classifier for multivariate Bernoulli models.

## Dependencies
    sklearn.naive_bayes.BernoulliNB

Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame for fitting the classifier
    alpha: float
        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    binarize: float
        Threshold for binarizing (mapping to booleans) of sample features.

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
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)

training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    BernoulliNB(alpha=0.96, binarize=0.86, fit_prior=True)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

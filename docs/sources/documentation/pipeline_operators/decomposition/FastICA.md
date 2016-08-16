# Fast algorithm for Independent Component Analysis
* * *

Uses Scikit-learn's FastICA to transform the feature set.

## Dependencies
    sklearn.decomposition.FastICA


Parameters
----------
    input_df: numpy.ndarray {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    tol: float
        Tolerance on update at each iteration.

Returns
-------
    modified_df: numpy.ndarray {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features


Example Exported Code
---------------------

```Python
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import FastICA
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
input_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1), input_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes =\
    train_test_split(features, tpot_data['class'], random_state=42)


exported_pipeline = make_pipeline(
    FastICA(tol=0.96),
    DecisionTreeClassifier(min_weight_fraction_leaf=0.5)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)

```

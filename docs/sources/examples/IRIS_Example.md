# IRIS Example

The following code illustrates the usage of TPOT with the IRIS data set.

```python
from tpot import TPOT
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
```

Running this code should discover a pipeline that achieves ~92% testing accuracy. Note that sometimes when both `train_size` and `test_size` aren't specified in `train_test_split()` calls, the split doesn't use the entire data set, so we need to specify both.

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

After running the above code, the corresponding Python code should be exported to the `tpot_iris_pipeline.py` file and look similar to the following:

```python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    ExtraTreesClassifier(criterion="gini", max_features=1.0, min_weight_fraction_leaf=0.5, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

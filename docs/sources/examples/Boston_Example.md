The following code illustrates the usage of TPOT with the Boston house prices data set.

```python
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```

Running this code should discover a pipeline that achieves about 12.77 mean squared error (MSE).

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

After running the above code, the corresponding Python code should be exported to the `tpot_boston_pipeline.py` file and look similar to the following:

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the target is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)

training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    ExtraTreesRegressor(max_features=0.76, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

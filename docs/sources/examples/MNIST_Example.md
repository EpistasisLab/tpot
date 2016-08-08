# MNIST Example

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves ~98% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```python
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
features = tpot_data.view((np.float64, len(tpot_data.dtype.names)))
features = np.delete(features, tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    make_union(VotingClassifier(estimators=[("clf", GradientBoostingClassifier(learning_rate=1.0, max_features=1.0, min_weight_fraction_leaf=1e-05, n_estimators=500))]), FunctionTransformer(lambda X: X)),
    ExtraTreesClassifier(criterion="entropy", max_features=0.1, min_weight_fraction_leaf=0.05, n_estimators=500)
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

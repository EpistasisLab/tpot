## Iris flower classification

The following code illustrates the usage of TPOT with the Iris data set, which is a simple supervised classification problem.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
```

Running this code should discover a pipeline that achieves about 97% testing accuracy.

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

After running the above code, the corresponding Python code should be exported to the `tpot_iris_pipeline.py` file and look similar to the following:

```Python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    Normalizer(),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

## MNIST digit recognition

Below is a minimal working example with the practice MNIST data set, which is an image classification problem.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

Running this code should discover a pipeline that achieves about 98% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```Python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = KNeighborsClassifier(n_neighbors=6, weights="distance")

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

## Boston housing prices modeling

The following code illustrates the usage of TPOT with the Boston housing prices data set, which is a regression problem.

```Python
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

digits = load_boston()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```

Running this code should discover a pipeline that achieves at least 10 mean squared error (MSE) on the test set.

For details on how the `fit()`, `score()` and `export()` functions work, see the [usage documentation](/using/).

After running the above code, the corresponding Python code should be exported to the `tpot_boston_pipeline.py` file and look similar to the following:

```Python
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",
                                              max_features=0.9, min_samples_leaf=5,
                                              min_samples_split=6)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

## Titanic survival analysis

To see the TPOT applied the Titanic Kaggle dataset, see the Jupyter notebook [here](https://github.com/rhiever/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb). This example shows how to take a messy dataset and preprocess it such that it can be used in scikit-learn and TPOT.

## Overview

The following sections illustrate the usage of TPOT with various datasets, each
belonging to a typical class of machine learning tasks.

| Dataset | Task                    | Task class             | Dataset description | Jupyter notebook                                                                           |
| ------- | ----------------------- | ---------------------- |:-------------------:|:------------------------------------------------------------------------------------------:|
| Iris                  | flower classification   | classification         | [link](https://archive.ics.uci.edu/ml/datasets/iris) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/IRIS.ipynb) |
| MNIST                 | digit recognition       | (image) classification | [link](https://yann.lecun.com/exdb/mnist/) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MNIST.ipynb) |
| Boston                | housing prices modeling | regression             | [link](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) | N/A    |
| Titanic               | survival analysis       | classification         | [link](https://www.kaggle.com/c/titanic/data) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb) |
| Bank Marketing        | subscription prediction | classification         | [link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Portuguese%20Bank%20Marketing/Portuguese%20Bank%20Marketing%20Stratergy.ipynb) |
| MAGIC Gamma Telescope | event detection         | classification         | [link](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MAGIC%20Gamma%20Telescope/MAGIC%20Gamma%20Telescope.ipynb) |

**Notes:**
- For details on how the `fit()`, `score()` and `export()` methods work, refer to the [usage documentation](/using/).
- Upon re-running the experiments, your resulting pipelines _may_ differ (to some extent) from the ones demonstrated here.

## Iris flower classification

The following code illustrates how TPOT can be employed for performing a simple _classification task_ over the Iris dataset.

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

Running this code should discover a pipeline (expored as `tpot_iris_pipeline.py`) that achieves about 97% test accuracy:

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
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=None)

exported_pipeline = make_pipeline(
    Normalizer(),
    GaussianNB()
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

## MNIST digit recognition

Below is a minimal working example with the practice MNIST dataset, which is an _image classification problem_.

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

Running this code should discover a pipeline (expored as `tpot_mnist_pipeline.py`) that achieves about 98% test accuracy:

```Python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=None)

exported_pipeline = KNeighborsClassifier(n_neighbors=6, weights="distance")

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

## Boston housing prices modeling

The following code illustrates how TPOT can be employed for performing a _regression task_ over the Boston housing prices dataset.

```Python
from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

housing = load_boston()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```

Running this code should discover a pipeline (exported as `tpot_boston_pipeline.py`) that achieves at least 10 mean squared error (MSE) on the test set:

```Python
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=None)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",
                                              max_features=0.9, min_samples_leaf=5,
                                              min_samples_split=6)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

## Titanic survival analysis

To see the TPOT applied the Titanic Kaggle dataset, see the Jupyter notebook [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb). This example shows how to take a messy dataset and preprocess it such that it can be used in scikit-learn and TPOT.

## Portuguese Bank Marketing

The corresponding Jupyter notebook, containing the associated data preprocessing and analysis, can be found [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Portuguese%20Bank%20Marketing/Portuguese%20Bank%20Marketing%20Stratergy.ipynb).

## MAGIC Gamma Telescope
The corresponding Jupyter notebook, containing the associated data preprocessing and analysis, can be found [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MAGIC%20Gamma%20Telescope/MAGIC%20Gamma%20Telescope.ipynb).

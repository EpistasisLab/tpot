# Overview

The following sections illustrate the usage of TPOT with various datasets, each
belonging to a typical class of machine learning tasks.

| Dataset | Task                    | Task class             | Dataset description | Jupyter notebook                                                                           |
| ------- | ----------------------- | ---------------------- |:-------------------:|:------------------------------------------------------------------------------------------:|
| Iris                  | flower classification   | classification         | [link](https://archive.ics.uci.edu/ml/datasets/iris) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/IRIS.ipynb) |
| Optical Recognition of Handwritten Digits                 | digit recognition       | (image) classification | [link](https://scikit-learn.org/stable/datasets/index.html#digits-dataset) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Digits.ipynb) |
| Boston                | housing prices modeling | regression             | [link](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) | N/A    |
| Titanic               | survival analysis       | classification         | [link](https://www.kaggle.com/c/titanic/data) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb) |
| Bank Marketing        | subscription prediction | classification         | [link](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Portuguese%20Bank%20Marketing/Portuguese%20Bank%20Marketing%20Strategy.ipynb) |
| MAGIC Gamma Telescope | event detection         | classification         | [link](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MAGIC%20Gamma%20Telescope/MAGIC%20Gamma%20Telescope.ipynb) |
| cuML Classification Example | random classification problem         | classification         | [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/cuML_Classification_Example.ipynb) |
| cuML Regression Example | random regression problem         | regression         | [link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html) | [link](https://github.com/EpistasisLab/tpot/blob/master/tutorials/cuML_Regression_Example.ipynb) |

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
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
```

Running this code should discover a pipeline (exported as `tpot_iris_pipeline.py`) that achieves about 97% test accuracy:

```Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9826086956521738
exported_pipeline = make_pipeline(
    Normalizer(norm="l2"),
    KNeighborsClassifier(n_neighbors=5, p=2, weights="distance")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

## Digits dataset

Below is a minimal working example with the optical recognition of handwritten digits dataset, which is an _image classification problem_.

```Python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')
```

Running this code should discover a pipeline (exported as `tpot_digits_pipeline.py`) that achieves about 98% test accuracy:

```Python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9799428471757372
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=LogisticRegression(C=0.1, dual=False, penalty="l1")),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=20, min_samples_split=19, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

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
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
```

Running this code should discover a pipeline (exported as `tpot_boston_pipeline.py`) that achieves at least 10 mean squared error (MSE) on the test set:

```Python
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -10.812040755234403
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.5, min_samples_leaf=2, min_samples_split=3, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
```

## Titanic survival analysis

To see the TPOT applied the Titanic Kaggle dataset, see the Jupyter notebook [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Titanic_Kaggle.ipynb). This example shows how to take a messy dataset and preprocess it such that it can be used in scikit-learn and TPOT.

## Portuguese Bank Marketing

The corresponding Jupyter notebook, containing the associated data preprocessing and analysis, can be found [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/Portuguese%20Bank%20Marketing/Portuguese%20Bank%20Marketing%20Stratergy.ipynb).

## MAGIC Gamma Telescope
The corresponding Jupyter notebook, containing the associated data preprocessing and analysis, can be found [here](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MAGIC%20Gamma%20Telescope/MAGIC%20Gamma%20Telescope.ipynb).

## Neural network classifier using TPOT-NN
By loading the <a href="https://github.com/EpistasisLab/tpot/blob/master/tpot/config/classifier_nn.py">TPOT-NN configuration dictionary</a>, PyTorch estimators will be included for classification. Users can also create their own NN configuration dictionary that includes `tpot.builtins.PytorchLRClassifier` and/or `tpot.builtins.PytorchMLPClassifier`, or they can specify them using a template string, as shown in the following example:

```Python
from tpot import TPOTClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=100, centers=2, n_features=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

clf = TPOTClassifier(config_dict='TPOT NN', template='Selector-Transformer-PytorchLRClassifier',
                     verbosity=2, population_size=10, generations=10)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
clf.export('tpot_nn_demo_pipeline.py')
```

This example is somewhat trivial, but it should result in nearly 100% classification accuracy.

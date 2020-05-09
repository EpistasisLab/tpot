Master status: [![Master Build Status - Mac/Linux](https://travis-ci.org/EpistasisLab/tpot.svg?branch=master)](https://travis-ci.org/EpistasisLab/tpot)
[![Master Build Status - Windows](https://ci.appveyor.com/api/projects/status/b7bmpwpkjhifrm7v/branch/master?svg=true)](https://ci.appveyor.com/project/weixuanfu/tpot?branch=master)
[![Master Coverage Status](https://coveralls.io/repos/github/EpistasisLab/tpot/badge.svg?branch=master)](https://coveralls.io/github/EpistasisLab/tpot?branch=master)

Development status: [![Development Build Status - Mac/Linux](https://travis-ci.org/EpistasisLab/tpot.svg?branch=development)](https://travis-ci.org/EpistasisLab/tpot/branches)
[![Development Build Status - Windows](https://ci.appveyor.com/api/projects/status/b7bmpwpkjhifrm7v/branch/development?svg=true)](https://ci.appveyor.com/project/weixuanfu/tpot?branch=development)
[![Development Coverage Status](https://coveralls.io/repos/github/EpistasisLab/tpot/badge.svg?branch=development)](https://coveralls.io/github/EpistasisLab/tpot?branch=development)

Package information: [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)
[![PyPI version](https://badge.fury.io/py/TPOT.svg)](https://badge.fury.io/py/TPOT)

<p align="center">
<img src="https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-logo.jpg" width=300 />
</p>

**TPOT** stands for **T**ree-based **P**ipeline **O**ptimization **T**ool. Consider TPOT your **Data Science Assistant**. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

![TPOT Demo](https://github.com/EpistasisLab/tpot/blob/master/images/tpot-demo.gif "TPOT Demo")

TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

![An example Machine Learning pipeline](https://github.com/EpistasisLab/tpot/blob/master/images/tpot-ml-pipeline.png "An example Machine Learning pipeline")

<p align="center"><strong>An example Machine Learning pipeline</strong></p>

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.

![An example TPOT pipeline](https://github.com/EpistasisLab/tpot/blob/master/images/tpot-pipeline-example.png "An example TPOT pipeline")

TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

**TPOT is still under active development** and we encourage you to check back on this repository regularly for updates.

For further information about TPOT, please see the [project documentation](http://epistasislab.github.io/tpot/).

## License

Please see the [repository license](https://github.com/EpistasisLab/tpot/blob/master/LICENSE) for the licensing and usage information for TPOT.

Generally, we have licensed TPOT to make it as widely usable as possible.

## Installation

We maintain the [TPOT installation instructions](http://epistasislab.github.io/tpot/installing/) in the documentation. TPOT requires a working installation of Python.

## Usage

TPOT can be used [on the command line](http://epistasislab.github.io/tpot/using/#tpot-on-the-command-line) or [with Python code](http://epistasislab.github.io/tpot/using/#tpot-with-code).

Click on the corresponding links to find more information on TPOT usage in the documentation.

## Examples

### Classification

Below is a minimal working example with the the optical recognition of handwritten digits dataset.

```python
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

Running this code should discover a pipeline that achieves about 98% testing accuracy, and the corresponding Python code should be exported to the `tpot_digits_pipeline.py` file and look similar to the following:

```python
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

### Regression

Similarly, TPOT can optimize pipelines for regression problems. Below is a minimal working example with the practice Boston housing prices data set.

```python
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

which should result in a pipeline that achieves about 12.77 mean squared error (MSE), and the Python code in `tpot_boston_pipeline.py` should look similar to:

```python
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

Check the documentation for [more examples and tutorials](http://epistasislab.github.io/tpot/examples/).

## Contributing to TPOT

We welcome you to [check the existing issues](https://github.com/EpistasisLab/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/EpistasisLab/tpot/issues/new) so we can discuss it.

Before submitting any contributions, please review our [contribution guidelines](http://epistasislab.github.io/tpot/contributing/).

## Having problems or have questions about TPOT?

Please [check the existing open and closed issues](https://github.com/EpistasisLab/tpot/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, [file a new issue](https://github.com/EpistasisLab/tpot/issues/new) on this repository so we can review your issue.

## Citing TPOT

If you use TPOT in a scientific publication, please consider citing at least one of the following papers:

Trang T. Le, Weixuan Fu and Jason H. Moore (2020). [Scaling tree-based automated machine learning to biomedical big data with a feature set selector](https://academic.oup.com/bioinformatics/article/36/1/250/5511404). *Bioinformatics*.36(1): 250-256.

BibTeX entry:

```bibtex
@article{le2020scaling,
  title={Scaling tree-based automated machine learning to biomedical big data with a feature set selector},
  author={Le, Trang T and Fu, Weixuan and Moore, Jason H},
  journal={Bioinformatics},
  volume={36},
  number={1},
  pages={250--256},
  year={2020},
  publisher={Oxford University Press}
}
```


Randal S. Olson, Ryan J. Urbanowicz, Peter C. Andrews, Nicole A. Lavender, La Creis Kidd, and Jason H. Moore (2016). [Automating biomedical data science through tree-based pipeline optimization](http://link.springer.com/chapter/10.1007/978-3-319-31204-0_9). *Applications of Evolutionary Computation*, pages 123-137.

BibTeX entry:

```bibtex
@inbook{Olson2016EvoBio,
    author={Olson, Randal S. and Urbanowicz, Ryan J. and Andrews, Peter C. and Lavender, Nicole A. and Kidd, La Creis and Moore, Jason H.},
    editor={Squillero, Giovanni and Burelli, Paolo},
    chapter={Automating Biomedical Data Science Through Tree-Based Pipeline Optimization},
    title={Applications of Evolutionary Computation: 19th European Conference, EvoApplications 2016, Porto, Portugal, March 30 -- April 1, 2016, Proceedings, Part I},
    year={2016},
    publisher={Springer International Publishing},
    pages={123--137},
    isbn={978-3-319-31204-0},
    doi={10.1007/978-3-319-31204-0_9},
    url={http://dx.doi.org/10.1007/978-3-319-31204-0_9}
}
```

Randal S. Olson, Nathan Bartley, Ryan J. Urbanowicz, and Jason H. Moore (2016). [Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science](http://dl.acm.org/citation.cfm?id=2908918). *Proceedings of GECCO 2016*, pages 485-492.

BibTeX entry:

```bibtex
@inproceedings{OlsonGECCO2016,
    author = {Olson, Randal S. and Bartley, Nathan and Urbanowicz, Ryan J. and Moore, Jason H.},
    title = {Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference 2016},
    series = {GECCO '16},
    year = {2016},
    isbn = {978-1-4503-4206-3},
    location = {Denver, Colorado, USA},
    pages = {485--492},
    numpages = {8},
    url = {http://doi.acm.org/10.1145/2908812.2908918},
    doi = {10.1145/2908812.2908918},
    acmid = {2908918},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```

Alternatively, you can cite the repository directly with the following DOI:

[![DOI](https://zenodo.org/badge/20747/rhiever/tpot.svg)](https://zenodo.org/badge/latestdoi/20747/rhiever/tpot)

## Support for TPOT

TPOT was developed in the [Computational Genetics Lab](http://epistasis.org/) at the [University of Pennsylvania](https://www.upenn.edu/) with funding from the [NIH](http://www.nih.gov/) under grant R01 AI117694. We are incredibly grateful for the support of the NIH and the University of Pennsylvania during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

[![Build Status](https://travis-ci.org/rhiever/tpot.svg?branch=master)](https://travis-ci.org/rhiever/tpot)
[![Code Health](https://landscape.io/github/rhiever/tpot/master/landscape.svg?style=flat)](https://landscape.io/github/rhiever/tpot/master)
[![Coverage Status](https://coveralls.io/repos/rhiever/tpot/badge.svg?branch=master&service=github)](https://coveralls.io/github/rhiever/tpot?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
[![PyPI version](https://badge.fury.io/py/tpot.svg)](https://badge.fury.io/py/tpot)

[![Join the chat at https://gitter.im/rhiever/tpot](https://badges.gitter.im/rhiever/tpot.svg)](https://gitter.im/rhiever/tpot?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<p align="center">
<img src="https://raw.githubusercontent.com/rhiever/tpot/master/images/tpot-logo.jpg" width=300 />
</p>

Consider TPOT your **Data Science Assistant**. TPOT is a Python tool that automatically creates and optimizes machine learning pipelines using genetic programming.

![TPOT Demo](https://github.com/rhiever/tpot/blob/master/images/tpot-demo.gif "TPOT Demo")

TPOT will automate the most tedious part of machine learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

![An example Machine Learning pipeline](https://github.com/rhiever/tpot/blob/master/images/tpot-ml-pipeline.png "An example Machine Learning pipeline")

<p align="center"><strong>An example Machine Learning pipeline</strong></p>

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.

![An example TPOT pipeline](https://github.com/rhiever/tpot/blob/master/images/tpot-pipeline-example.png "An example TPOT pipeline")

TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

**TPOT is still under active development** and we encourage you to check back on this repository regularly for updates.

For further information about TPOT, please see the [project documentation](http://rhiever.github.io/tpot/).

## License

Please see the [repository license](https://github.com/rhiever/tpot/blob/master/LICENSE) for the licensing and usage information for TPOT.

Generally, we have licensed TPOT to make it as widely usable as possible.

## Installation

We maintain the [TPOT installation instructions](http://rhiever.github.io/tpot/installing/) in the documentation. TPOT requires a working installation of Python.

## Usage

TPOT can be used [on the command line](http://rhiever.github.io/tpot/using/#tpot-on-the-command-line) or [with Python code](http://rhiever.github.io/tpot/using/#tpot-with-code).

Click on the corresponding links to find more information on TPOT usage in the documentation.

## Examples

### Classification

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves about 98% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1),
                     tpot_data.dtype.names.index('class'), axis=1)

training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    KNeighborsClassifier(n_neighbors=3, weights="uniform")
)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

### Regression

Similarly, TPOT can optimize pipelines for regression problems. Below is a minimal working example with the practice Boston housing prices data set.

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

which should result in a pipeline that achieves about 12.77 mean squared error (MSE), and the Python code in `tpot_boston_pipeline.py` should look similar to:

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

Check the documentation for [more examples and tutorials](http://rhiever.github.io/tpot/examples/MNIST_Example/).

## Contributing to TPOT

We welcome you to [check the existing issues](https://github.com/rhiever/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/rhiever/tpot/issues/new) so we can discuss it.

Before submitting any contributions, please review our [contribution guidelines](http://rhiever.github.io/tpot/contributing/).

## Having problems or have questions about TPOT?

Please [check the existing open and closed issues](https://github.com/rhiever/tpot/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, [file a new issue](https://github.com/rhiever/tpot/issues/new) on this repository so we can review your issue.

## Citing TPOT

If you use TPOT in a scientific publication, please consider citing at least one of the following papers:

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

Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science

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

TPOT was developed in the [Computational Genetics Lab](http://epistasis.org) with funding from the [NIH](http://www.nih.gov). We're incredibly grateful for their support during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

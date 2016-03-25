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

## Example

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
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a random forest classifier
rfc1 = RandomForestClassifier(n_estimators=200, max_features=min(64, len(result1.columns) - 1))
rfc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['rfc1-classification'] = rfc1.predict(result1.drop('class', axis=1).values)
```

## Contributing to TPOT

We welcome you to [check the existing issues](https://github.com/rhiever/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/rhiever/tpot/issues/new) so we can discuss it.

Before submitting any contributions, please review our [contribution guidelines](http://rhiever.github.io/tpot/contributing/).

## Having problems or have questions about TPOT?

Please [check the existing open and closed issues](https://github.com/rhiever/tpot/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, [file a new issue](https://github.com/rhiever/tpot/issues/new) on this repository so we can review your issue.

## Citing TPOT

If you use TPOT in a scientific publication, please consider citing at least one of the following papers:

R. S. Olson et al. [Automating biomedical data science through tree-based pipeline optimization](http://arxiv.org/abs/1601.07925). In G. Squillero and P. Burelli, editors, *Proceedings of the 18th European Conference on the Applications of Evolutionary and Bio-inspired Computation*, Lecture Notes in Computer Science, Berlin, Germany, 2016. Springer-Verlag.

BibTeX entry:

```bibtex
@inproceedings{Olson2016EvoBIO,
author = {Olson, Randal S. and Urbanowicz, Ryan J. and Andrews, Peter C. and Lavender, Nicole A. and Kidd, La Creis and Moore, Jason H.},
title = {Automating biomedical data science through tree-based pipeline optimization},
booktitle = {Proceedings of the 18th European Conference on the Applications of Evolutionary and Bio-inspired Computation},
series = {Lecture Notes in Computer Science},
year = {2016},
location = {Porto, Portugal},
numpages = {16},
editor = {Squillero, G and Burelli, P},
publisher = {Springer-Verlag},
address = {Berlin, Germany}
}
```

Alternatively, you can cite the repository directly with the following DOI:

[![DOI](https://zenodo.org/badge/20747/rhiever/tpot.svg)](https://zenodo.org/badge/latestdoi/20747/rhiever/tpot)

## Support for TPOT

TPOT was developed in the [Computational Genetics Lab](http://epistasis.org) with funding from the [NIH](http://www.nih.gov). We're incredibly grateful for their support during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

[![Build Status](https://travis-ci.org/rhiever/tpot.svg?branch=master)](https://travis-ci.org/rhiever/tpot)
[![Code Health](https://landscape.io/github/rhiever/tpot/master/landscape.svg?style=flat)](https://landscape.io/github/rhiever/tpot/master)
[![Coverage Status](https://coveralls.io/repos/rhiever/tpot/badge.svg?branch=master&service=github)](https://coveralls.io/github/rhiever/tpot?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)

# Tree-based Pipeline Optimization Tool (TPOT)

Consider TPOT your **Data Science Assistant**. TPOT is a Python tool that automatically creates and optimizes Machine Learning pipelines using genetic programming.

TPOT will automate the most tedious part of Machine Learning by intelligently exploring thousands of possible pipelines to find the best one for your data.

![An example Machine Learning pipeline](https://github.com/rhiever/tpot/blob/master/images/tpot-ml-pipeline.png "An example Machine Learning pipeline")

<p align="center"><strong>An example Machine Learning pipeline</strong></p>

Once TPOT is finished searching (or you get tired of waiting), it provides you with the Python code for the best pipeline it found so you can tinker with the pipeline from there.

![An example TPOT pipeline](https://github.com/rhiever/tpot/blob/master/images/tpot-pipeline-example.png "An example TPOT pipeline")

TPOT is built on top of scikit-learn, so all of the code it generates should look familiar... if you're familiar with scikit-learn, anyway.

**TPOT is still under active development** and we encourage you to check back on this repository regularly for updates.

## License

Please see the [repository license](https://github.com/rhiever/tpot/blob/master/LICENSE) for the licensing and usage information for TPOT.

## Installation

TPOT is built on top of several existing Python libraries, including:

* NumPy

* SciPy

* pandas

* scikit-learn

* DEAP

Except for DEAP, all of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use of Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, pandas, and scikit-learn can be installed in Anaconda via the command:

```Shell
conda install numpy scipy pandas scikit-learn
```

DEAP can be installed with `pip` via the command:

```Shell
pip install deap
```

Finally to install TPOT, run the following command:

```Shell
pip install tpot
```

Please [file a new issue](https://github.com/rhiever/tpot/issues/new) if you run into installation problems.

## Usage

TPOT can be used in two ways: via code and via the command line. We will eventually develop a GUI for TPOT.

### Using TPOT via code

We've taken care to design the TPOT interface to be as similar as possible to scikit-learn.

TPOT can be imported just like any regular Python module. To import TPOT, type:

```Python
from tpot import TPOT
```

then create an instance of TPOT as follows:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT()
```

Note that you can pass several parameters to the TPOT instantiation call:

* `generations`: The number of generations to run pipeline optimization for. Must be > 0. The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
* `population_size`: The number of pipelines in the genetic algorithm population. Must be > 0. The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
* `mutation_rate`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `crossover_rate`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `random_state`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `verbosity`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all

Some example code with custom TPOT parameters might look like:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
```

Now TPOT is ready to work! You can tell TPOT to optimize a pipeline based on a data set with the `fit` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
```

then evaluate the final pipeline with the `score()` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(training_features, training_classes, testing_features, testing_classes))
```

Note that you need to pass the training data to the `score()` function so the pipeline re-trains the scikit-learn models on the training data.

Finally, you can tell TPOT to export the optimized pipeline to a text file with the `export()` function:

```Python
from tpot import TPOT

pipeline_optimizer = TPOT(generations=100, random_state=42, verbosity=2)
pipeline_optimizer.fit(training_features, training_classes)
print(pipeline_optimizer.score(training_features, training_classes, testing_features, testing_classes))
pipeline_optimizer.export('tpot_exported_pipeline.py')
```

Once this code finishes running, `tpot_exported_pipeline.py` will contain the Python code for the optimized pipeline.

### Using TPOT via the command line

To use TPOT via the command line, enter the following command to see the parameters that TPOT can receive:

```Shell
tpot --help
```

The following parameters will display along with their descriptions:

* `-i` / `INPUT_FILE`: The path to the data file to optimize the pipeline on. Make sure that the class column in the file is labeled as "class".
* `-is` / `INPUT_SEPARATOR`: The character used to separate columns in the input file. Commas (,) and tabs (\t) are the most common separators.
* `-o` / `OUTPUT_FILE`: The path to a file that you wish to export the pipeline code into. By default, exporting is disabled.
* `-g` / `GENERATIONS`: The number of generations to run pipeline optimization for. Must be > 0. The more generations you give TPOT to run, the longer it takes, but it's also more likely to find better pipelines.
* `-p` / `POPULATION`: The number of pipelines in the genetic algorithm population. Must be > 0. The more pipelines in the population, the slower TPOT will run, but it's also more likely to find better pipelines.
* `-mr` / `MUTATION_RATE`: The mutation rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to apply random changes to every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-xr` / `CROSSOVER_RATE`: The crossover rate for the genetic programming algorithm in the range [0.0, 1.0]. This tells the genetic programming algorithm how many pipelines to "breed" every generation. We don't recommend that you tweak this parameter unless you know what you're doing.
* `-s` / `RANDOM_STATE`: The random number generator seed for TPOT. Use this to make sure that TPOT will give you the same results each time you run it against the same data set with that seed.
* `-v` / `VERBOSITY`: How much information TPOT communicates while it's running. 0 = none, 1 = minimal, 2 = all

An example command-line call to TPOT may look like:

```Shell
tpot -i data/mnist.csv -is , -o tpot_exported_pipeline.py -g 100 -s 42 -v 2
```

## Examples

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75)

tpot = TPOT(generations=5)
tpot.fit(X_train, y_train)
print(tpot.score(X_train, y_train, X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')
```

Running this code should discover a pipeline that achieves ~97% testing accuracy, and the corresponding Python code should be exported to the `tpot_mnist_pipeline.py` file and look similar to the following:

```python
import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indeces, testing_indeces = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75)))


result1 = tpot_data.copy()

# Perform classification with a logistic regression classifier
lrc1 = LogisticRegression(C=2.8214285714285716)
lrc1.fit(result1.loc[training_indeces].drop('class', axis=1).values, result1.loc[training_indeces, 'class'].values)
result1['lrc1-classification'] = lrc1.predict(result1.drop('class', axis=1).values)
```

## Want to get involved with TPOT?

We welcome you to [check the existing issues](https://github.com/rhiever/tpot/issues/) for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please [file a new issue](https://github.com/rhiever/tpot/issues/new) so we can discuss it.

## Having problems or have questions about TPOT?

Please [check the existing open and closed issues](https://github.com/rhiever/tpot/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, please [file a new issue](https://github.com/rhiever/tpot/issues/new) on this repository so we can review your issue.

## Support for TPOT

TPOT was developed in the [Computational Genetics Lab](http://epistasis.org) with funding from the [NIH](http://www.nih.gov). We're incredibly grateful for their support during the development of this project!

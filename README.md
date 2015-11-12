# Tree-based Pipeline Optimization Tool (TPOT)

A Python tool that automatically creates and optimizes Machine Learning pipelines using genetic programming.

## License

Please see the [repository license](https://github.com/rhiever/tpot/blob/master/LICENSE) for the licensing and usage information for this library.

## Installation

TPOT is built on top of several existing Python libraries, including:

* NumPy

* pandas

* scikit-learn

* DEAP

Except for DEAP, all of the necessary Python packages can be installed via the [Anaconda Python installer](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use of Python 3 over Python 2 if you're given the choice.

DEAP can be installed with `pip` via the command:

```shell
pip install deap
```

**If you don't care about the details and just want to install TPOT, run the following command:**

```shell
pip install tpot
```

`pip` should be able to sort out all of the dependencies for you.

## Usage

Below is a minimal working example with the practice MNIST data set.

```python
from tpot import TPOT
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.75)

tpot = TPOT(generations=5)
tpot.optimize(X_train, y_train)
tpot.score(X_test, y_test)
```

Running this code should discover a pipeline that achieves >=98% testing accuracy.

## Having problems or have questions about TPOT?

Please [check the existing open and closed issues](https://github.com/rhiever/tpot/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, please [file a new issue](https://github.com/rhiever/tpot/issues/new) on this repository so we can review your issue.

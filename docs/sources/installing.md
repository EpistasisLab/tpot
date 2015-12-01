# Installation

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

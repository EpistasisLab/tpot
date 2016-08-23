TPOT is built on top of several existing Python libraries, including:

* NumPy

* SciPy

* scikit-learn

* DEAP

* update_checker

* tqdm

Most of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use of Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```Shell
conda install numpy scipy scikit-learn
```

DEAP, update_checker, and tqdm (used for verbose TPOT runs) can be installed with `pip` via the command:

```Shell
pip install deap update_checker tqdm
```

Finally to install TPOT itself, run the following command:

```Shell
pip install tpot
```

Please [file a new issue](https://github.com/rhiever/tpot/issues/new) if you run into installation problems.

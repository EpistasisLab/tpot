TPOT is built on top of several existing Python libraries, including:

* NumPy

* SciPy

* scikit-learn

* DEAP

* update_checker

* tqdm

* Pathos

Most of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use of Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```Shell
conda install numpy scipy scikit-learn
```

DEAP, update_checker, tqdm (used for verbose TPOT runs), and Pathos can be installed with `pip` via the command:

```Shell
pip install deap update_checker tqdm pathos
```

**For the Windows OS**, the pywin32 module is required if the Python is NOT installed via [Anaconda Python distribution](https://www.continuum.io/downloads) and can be installed with `pip` via the command:

```Shell
pip install pywin32
```

**Optionally**, install XGBoost if you would like TPOT to use XGBoost. XGBoost is entirely optional, and TPOT will still function normally without XGBoost if you do not have it installed.

```Shell
pip install xgboost
```

If you have issues installing XGBoost, check the [XGBoost installation documentation](http://xgboost.readthedocs.io/en/latest/build.html).

Finally to install TPOT itself, run the following command:

```Shell
pip install tpot
```

Please [file a new issue](https://github.com/rhiever/tpot/issues/new) if you run into installation problems.

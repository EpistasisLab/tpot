# Installation

TPOT is built on top of several existing Python libraries, including:

* [NumPy](http://www.numpy.org/)

* [SciPy](https://www.scipy.org/)

* [scikit-learn](http://www.scikit-learn.org/)

* [DEAP](https://github.com/DEAP/deap)

* [update_checker](https://github.com/bboe/update_checker)

* [tqdm](https://github.com/tqdm/tqdm)

* [stopit](https://github.com/glenfant/stopit)

* [pandas](http://pandas.pydata.org)

* [joblib](https://joblib.readthedocs.io/en/latest/)

Most of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use of Python 3 over Python 2 if you're given the choice.

You can install TPOT using `pip` or `conda-forge`.

## pip

NumPy, SciPy, scikit-learn, pandas, joblib, and PyTorch can be installed in Anaconda via the command:

```Shell
conda install numpy scipy scikit-learn pandas joblib pytorch
```

DEAP, update_checker, tqdm and stopit can be installed with `pip` via the command:

```Shell
pip install deap update_checker tqdm stopit
```

**Optionally**, you can install [XGBoost](https://github.com/dmlc/xgboost) if you would like TPOT to use the eXtreme Gradient Boosting models. XGBoost is entirely optional, and TPOT will still function normally without XGBoost if you do not have it installed. **Windows users: pip installation may not work on some Windows environments, and it may cause unexpected errors.**

```Shell
pip install xgboost
```

If you have issues installing XGBoost, check the [XGBoost installation documentation](http://xgboost.readthedocs.io/en/latest/build.html).

If you plan to use [Dask](http://dask.pydata.org/en/latest/) for parallel training, make sure to install [dask[delay] and dask[dataframe]](https://docs.dask.org/en/latest/install.html) and [dask_ml](https://dask-ml.readthedocs.io/en/latest/install.html).

```Shell
pip install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3
```

If you plan to use the [TPOT-MDR configuration](https://arxiv.org/abs/1702.01780), make sure to install [scikit-mdr](https://github.com/EpistasisLab/scikit-mdr) and [scikit-rebate](https://github.com/EpistasisLab/scikit-rebate):

```Shell
pip install scikit-mdr skrebate
```

To enable support for [PyTorch](https://pytorch.org/)-based neural networks (TPOT-NN), you will need to install PyTorch. TPOT-NN will work with either CPU or GPU PyTorch, but we strongly recommend using a GPU version, if possible, as CPU PyTorch models tend to train very slowly.

We recommend following [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) customized for your operating system and Python distribution.

Finally to install TPOT itself, run the following command:

```Shell
pip install tpot
```

## conda-forge

To install tpot and its core dependencies you can use:

```Shell
conda install -c conda-forge tpot
```

To install additional dependencies you can use:

```Shell
conda install -c conda-forge tpot xgboost dask dask-ml scikit-mdr skrebate
```

As mentioned above, we recommend following [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) for installing it to enable support for [PyTorch](https://pytorch.org/)-based neural networks (TPOT-NN).

## Installation problems

Please [file a new issue](https://github.com/EpistasisLab/tpot/issues/new) if you run into installation problems.

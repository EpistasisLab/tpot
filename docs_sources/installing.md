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

* [xgboost](https://xgboost.readthedocs.io/en/latest/)

Most of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.anaconda.com/products/individual), which we strongly recommend that you use. **Support for Python 3.4 and below has been officially dropped since version 0.11.0.**


You can install TPOT using `pip` or `conda-forge`.

## pip

NumPy, SciPy, scikit-learn, pandas, joblib, and PyTorch can be installed in Anaconda via the command:

```Shell
conda install numpy scipy scikit-learn pandas joblib pytorch
```

DEAP, update_checker, tqdm, stopit and xgboost can be installed with `pip` via the command:

```Shell
pip install deap update_checker tqdm stopit xgboost
```

**Windows users: pip installation may not work on some Windows environments, and it may cause unexpected errors.** If you have issues installing XGBoost, check the [XGBoost installation documentation](http://xgboost.readthedocs.io/en/latest/build.html).

If you plan to use [Dask](http://dask.pydata.org/en/latest/) for parallel training, make sure to install [dask[delay] and dask[dataframe]](https://docs.dask.org/en/latest/install.html) and [dask_ml](https://dask-ml.readthedocs.io/en/latest/install.html). **It is noted that dask-ml>=1.7 requires distributed>=2.4.0 and scikit-learn>=0.23.0.**

```Shell
pip install dask[delayed] dask[dataframe] dask-ml fsspec>=0.3.3 distributed>=2.10.0
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

## Installation for using TPOT-cuML configuration

With "TPOT cuML" configuration (see <a href="../using/#built-in-tpot-configurations">built-in configurations</a>), TPOT will search over a restricted configuration using the GPU-accelerated estimators in [RAPIDS cuML](https://github.com/rapidsai/cuml) and [DMLC XGBoost](https://github.com/dmlc/xgboost). **This configuration requires an NVIDIA Pascal architecture or better GPU with [compute capability 6.0+](https://developer.nvidia.com/cuda-gpus), and that the library cuML is installed.** With this configuration, all model training and predicting will be GPU-accelerated. This configuration is particularly useful for medium-sized and larger datasets on which CPU-based estimators are a common bottleneck, and works for both the `TPOTClassifier` and `TPOTRegressor`.

Please download this conda environment <a href="https://github.com/EpistasisLab/tpot/blob/master/tpot-cuml.yml">yml file</a></td> to install TPOT for using TPOT-cuML configuration.

```
conda env create -f tpot-cuml.yml -n tpot-cuml
conda activate tpot-cuml
```


## Installation problems

Please [file a new issue](https://github.com/EpistasisLab/tpot/issues/new) if you run into installation problems.

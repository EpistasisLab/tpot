# TPOT2 ALPHA

![Tests](https://github.com/EpistasisLab/tpot2/actions/workflows/tests.yml/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tpot2?label=pypi%20downloads)](https://pypi.org/project/TPOT2)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/tpot2?label=conda%20downloads)](https://anaconda.org/conda-forge/tpot2)

TPOT stands for Tree-based Pipeline Optimization Tool. TPOT2 is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. Consider TPOT2 your Data Science Assistant.

TPOT2 is a rewrite of TPOT with some additional functionality. Notably, we added support for graph-based pipelines and additional parameters to better specify the desired search space. 
TPOT2 is currently in Alpha. This means that there will likely be some backwards incompatible changes to the API as we develop. Some implemented features may be buggy. There is a list of known issues written at the bottom of this README. Some features have placeholder names or are listed as "Experimental" in the doc string. These are features that may not be fully implemented and may or may not work with all other features.

If you are interested in using the current stable release of TPOT, you can do that here: [https://github.com/EpistasisLab/tpot/](https://github.com/EpistasisLab/tpot/). 


## License

Please see the [repository license](https://github.com/EpistasisLab/tpot2/blob/main/LICENSE) for the licensing and usage information for TPOT2.
Generally, we have licensed TPOT2 to make it as widely usable as possible.

## Documentation

[The documentation webpage can be found here.](https://epistasislab.github.io/tpot2/)

We also recommend looking at the Tutorials folder for jupyter notebooks with examples and guides.

## Installation

TPOT2 requires a working installation of Python.

### Creating a conda environment (optional)

We recommend using conda environments for installing TPOT2, though it would work equally well if manually installed without it.

[More information on making anaconda environments found here.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda create --name tpot2env python=3.10
conda activate tpot2env
```

### Packages Used

python version <3.12
numpy
scipy
scikit-learn
update_checker
tqdm
stopit
pandas
joblib
xgboost
matplotlib
traitlets
lightgbm
optuna
baikal
jupyter
networkx>
dask
distributed
dask-ml
dask-jobqueue
func_timeout
configspace

Many of the hyperparameter ranges used in our configspaces were adapted from either the original TPOT package or the AutoSklearn package. 

### Note for M1 Mac or other Arm-based CPU users

You need to install the lightgbm package directly from conda using the following command before installing TPOT2. 

This is to ensure that you get the version that is compatible with your system.

```
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
```

### Installing Extra Features with pip

If you want to utilize the additional features provided by TPOT2 along with `scikit-learn` extensions, you can install them using `pip`. The command to install TPOT2 with these extra features is as follows:

```
pip install tpot2[sklearnex]
```

Please note that while these extensions can speed up scikit-learn packages, there are some important considerations:

These extensions may not be fully developed and tested on Arm-based CPUs, such as M1 Macs. You might encounter compatibility issues or reduced performance on such systems.

We recommend using Python 3.9 when installing these extra features, as it provides better compatibility and stability.


### Developer/Latest Branch Installation


```
pip install -e /path/to/tpot2repo
```

If you downloaded with git pull, then the repository folder will be named TPOT2. (Note: this folder is the one that includes setup.py inside of it and not the folder of the same name inside it).
If you downloaded as a zip, the folder may be called tpot2-main. 


## Usage 

See the Tutorials Folder for more instructions and examples.

### Best Practices

#### 1 
TPOT2 uses dask for parallel processing. When Python is parallelized, each module is imported within each processes. Therefore it is important to protect all code within a `if __name__ == "__main__"` when running TPOT2 from a script. This is not required when running TPOT2 from a notebook.

For example:

```
#my_analysis.py

import tpot2
if __name__ == "__main__":
    X, y = load_my_data()
    est = tpot2.TPOTClassifier()
    est.fit(X,y)
    #rest of analysis
```

#### 2

When designing custom objective functions, avoid the use of global variables.

Don't Do:
```
global_X = [[1,2],[4,5]]
global_y = [0,1]
def foo(est):
    return my_scorer(est, X=global_X, y=global_y)

```

Instead use a partial

```
from functools import partial

def foo_scorer(est, X, y):
    return my_scorer(est, X, y)

if __name__=='__main__':
    X = [[1,2],[4,5]]
    y = [0,1]
    final_scorer = partial(foo_scorer, X=X, y=y)
```

Similarly when using lambda functions.

Dont Do:

```
def new_objective(est, a, b)
    #definition

a = 100
b = 20
bad_function = lambda est :  new_objective(est=est, a=a, b=b)
```

Do:
```
def new_objective(est, a, b)
    #definition

a = 100
b = 20
good_function = lambda est, a=a, b=b : new_objective(est=est, a=a, b=b)
```

### Tips

TPOT2 will not check if your data is correctly formatted. It will assume that you have passed in operators that can handle the type of data that was passed in. For instance, if you pass in a pandas dataframe with categorical features and missing data, then you should also include in your configuration operators that can handle those feautures of the data. Alternatively, if you pass in `preprocessing = True`, TPOT2 will impute missing values, one hot encode categorical features, then standardize the data. (Note that this is currently fitted and transformed on the entire training set before splitting for CV. Later there will be an option to apply per fold, and have the parameters be learnable.)


Setting `verbose` to 5 can be helpful during debugging as it will print out the error generated by failing pipelines. 


## Contributing to TPOT2

We welcome you to check the existing issues for bugs or enhancements to work on. If you have an idea for an extension to TPOT2, please file a new issue so we can discuss it.


### Support for TPOT2

TPOT2 was developed in the [Artificial Intelligence Innovation (A2I) Lab](http://epistasis.org/) at Cedars-Sinai with funding from the [NIH](http://www.nih.gov/) under grants U01 AG066833 and R01 LM010098. We are incredibly grateful for the support of the NIH and the Cedars-Sinai during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

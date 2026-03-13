# TPOT

<center>
<img src="https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-logo.jpg" width=300 />
</center>

<br>

![Tests](https://github.com/EpistasisLab/tpot/actions/workflows/tests.yml/badge.svg)
[![PyPI Downloads](https://img.shields.io/pypi/dm/tpot?label=pypi%20downloads)](https://pypi.org/project/TPOT)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/tpot?label=conda%20downloads)](https://anaconda.org/conda-forge/tpot)

TPOT stands for Tree-based Pipeline Optimization Tool. TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming. Consider TPOT your Data Science Assistant.

## Contributors

TPOT recently went through a major refactoring. The package was rewritten from scratch to improve efficiency and performance, support new features, and fix numerous bugs. New features include genetic feature selection, a significantly expanded and more flexible method of defining search spaces, multi-objective optimization, a more modular framework allowing for easier customization of the evolutionary algorithm, and more. While in development, this new version was referred to as "TPOT2" but we have now merged what was once TPOT2 into the main TPOT package. You can learn more about this new version of TPOT in our GPTP paper titled "TPOT2: A New Graph-Based Implementation of the Tree-Based Pipeline Optimization Tool for Automated Machine Learning."

    Ribeiro, P. et al. (2024). TPOT2: A New Graph-Based Implementation of the Tree-Based Pipeline Optimization Tool for Automated Machine Learning. In: Winkler, S., Trujillo, L., Ofria, C., Hu, T. (eds) Genetic Programming Theory and Practice XX. Genetic and Evolutionary Computation. Springer, Singapore. https://doi.org/10.1007/978-981-99-8413-8_1

The current version of TPOT was developed at Cedars-Sinai by:  
    - Pedro Henrique Ribeiro (Lead developer - https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)  
    - Anil Saini (anil.saini@cshs.org)  
    - Jose Hernandez (jgh9094@gmail.com)  
    - Jay Moran (jay.moran@cshs.org)  
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)  
    - Hyunjun Choi (hyunjun.choi@cshs.org)  
    - Gabriel Ketron (gabriel.ketron@cshs.org)  
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)  
    - Jason Moore (moorejh28@gmail.com)  

The original version of TPOT was primarily developed at the University of Pennsylvania by:  
    - Randal S. Olson (rso@randalolson.com)  
    - Weixuan Fu (weixuanf@upenn.edu)  
    - Daniel Angell (dpa34@drexel.edu)  
    - Jason Moore (moorejh28@gmail.com)  
    - and many more generous open-source contributors  

## License

Please see the [repository license](https://github.com/EpistasisLab/tpot/blob/main/LICENSE) for the licensing and usage information for TPOT.
Generally, we have licensed TPOT to make it as widely usable as possible.

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

## Documentation

[The documentation webpage can be found here.](https://epistasislab.github.io/tpot/)

We also recommend looking at the Tutorials folder for jupyter notebooks with examples and guides.

## Installation

TPOT requires a working installation of Python.

### Creating a conda environment (optional)

We recommend using conda environments for installing TPOT, though it would work equally well if manually installed without it.

[More information on making anaconda environments found here.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda create --name tpotenv python=3.10
conda activate tpotenv
```

### Packages Used

python version >=3.10, <3.14
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
jupyter
networkx
dask
distributed
dask-ml
dask-jobqueue
func_timeout
configspace

Many of the hyperparameter ranges used in our configspaces were adapted from either the original TPOT package or the AutoSklearn package. 

### Note for M1 Mac or other Arm-based CPU users

You need to install the lightgbm package directly from conda using the following command before installing TPOT. 

This is to ensure that you get the version that is compatible with your system.

```
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
```

### Installing Extra Features with pip

If you want to utilize the additional features provided by TPOT along with `scikit-learn` extensions, you can install them using `pip`. The command to install TPOT with these extra features is as follows:

```
pip install tpot[sklearnex]
```

Please note that while these extensions can speed up scikit-learn packages, there are some important considerations:

These extensions may not be fully developed and tested on Arm-based CPUs, such as M1 Macs. You might encounter compatibility issues or reduced performance on such systems.

We recommend using Python 3.9 when installing these extra features, as it provides better compatibility and stability.


### Developer/Latest Branch Installation


```
pip install -e /path/to/tpotrepo
```

If you downloaded with git pull, then the repository folder will be named TPOT. (Note: this folder is the one that includes setup.py inside of it and not the folder of the same name inside it).
If you downloaded as a zip, the folder may be called tpot-main. 


## Usage 

See the Tutorials Folder for more instructions and examples.

### Best Practices

#### 1 
TPOT uses dask for parallel processing. When Python is parallelized, each module is imported within each processes. Therefore it is important to protect all code within a `if __name__ == "__main__"` when running TPOT from a script. This is not required when running TPOT from a notebook.

For example:

```
#my_analysis.py

import tpot
if __name__ == "__main__":
    X, y = load_my_data()
    est = tpot.TPOTClassifier()
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

TPOT will not check if your data is correctly formatted. It will assume that you have passed in operators that can handle the type of data that was passed in. For instance, if you pass in a pandas dataframe with categorical features and missing data, then you should also include in your configuration operators that can handle those feautures of the data. Alternatively, if you pass in `preprocessing = True`, TPOT will impute missing values, one hot encode categorical features, then standardize the data. (Note that this is currently fitted and transformed on the entire training set before splitting for CV. Later there will be an option to apply per fold, and have the parameters be learnable.)


Setting `verbose` to 5 can be helpful during debugging as it will print out the error generated by failing pipelines. 


## Contributing to TPOT

We welcome you to check the existing issues for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please file a new issue so we can discuss it.

## Citing TPOT

If you use TPOT in a scientific publication, please consider citing at least one of the following papers:

Hernandez, J. G., Saini, A. K., Ghosh, A., & Moore, J. H. (2025). [The tree-based pipeline optimization tool: Tackling biomedical research problems with genetic programming and automated machine learning](https://www.cell.com/patterns/fulltext/S2666-3899(25)00162-X). Patterns, 6(7).

BibTeX entry:

```bibtext
@article{hernandez2025tree,
  title={The tree-based pipeline optimization tool: Tackling biomedical research problems with genetic programming and automated machine learning},
  author={Hernandez, Jose Guadalupe and Saini, Anil Kumar and Ghosh, Attri and Moore, Jason H},
  journal={Patterns},
  volume={6},
  number={7},
  year={2025},
  publisher={Elsevier}
}
```

Ribeiro, P., Saini, A., Moran, J., Matsumoto, N., Choi, H., Hernandez, M., & Moore, J. H. (2024). [TPOT2: A New Graph-Based Implementation of the Tree-Based Pipeline Optimization Tool for Automated Machine Learning](https://link.springer.com/chapter/10.1007/978-981-99-8413-8_1). In Genetic programming theory and practice XX (pp. 1-17). Singapore: Springer Nature Singapore.

BitTex entry:

```bibtex
@incollection{ribeiro2024tpot2,
  title={TPOT2: A New Graph-Based Implementation of the Tree-Based Pipeline Optimization Tool for Automated Machine Learning},
  author={Ribeiro, Pedro and Saini, Anil and Moran, Jay and Matsumoto, Nicholas and Choi, Hyunjun and Hernandez, Miguel and Moore, Jason H},
  booktitle={Genetic programming theory and practice XX},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```

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

## Related Papers

Trang T. Le, Weixuan Fu and Jason H. Moore (2020). [Scaling tree-based automated machine learning to biomedical big data with a feature set selector](https://academic.oup.com/bioinformatics/article/36/1/250/5511404). *Bioinformatics*.36(1): 250-256.

BibTeX entry:

```bibtex
@article{le2020scaling,
  title={Scaling tree-based automated machine learning to biomedical big data with a feature set selector},
  author={Le, Trang T and Fu, Weixuan and Moore, Jason H},
  journal={Bioinformatics},
  volume={36},
  number={1},
  pages={250--256},
  year={2020},
  publisher={Oxford University Press}
}
```


## Support for TPOT

TPOT was developed in the [Artificial Intelligence Innovation (A2I) Lab](http://epistasis.org/) at Cedars-Sinai with funding from the [NIH](http://www.nih.gov/) under grants U01 AG066833 and R01 LM010098. We are incredibly grateful for the support of the NIH and the Cedars-Sinai during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

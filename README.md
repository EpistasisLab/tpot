# TPOT2 ALPHA

![Tests](https://github.com/jay-m-dev/tpot2/actions/workflows/tests.yml/badge.svg)

TPOT2 is a rewrite of TPOT with some additional functionality. Notably, we added support for graph-based pipelines and additional parameters to better specify the desired search space. 
TPOT2 is currently in Alpha. This means that there will likely be some backwards incompatible changes to the API as we develop. Some implemented features may be buggy. There is a list of known issues written at the bottom of this README. Some features have placeholder names or are listed as "Experimental" in the doc string. These are features that may not be fully implemented and may or may work with all other features.

If you are interested in using the current stable release of TPOT, you can do that here: [https://github.com/EpistasisLab/tpot/](https://github.com/EpistasisLab/tpot/). 


## License

Please see the [repository license](https://github.com/EpistasisLab/tpot/blob/master/LICENSE) for the licensing and usage information for TPOT2.
Generally, we have licensed TPOT2 to make it as widely usable as possible.


## Installation

TPOT2 requires a working installation of Python.

### Creating a conda environment (optional)

We recommend using conda environments for installing TPOT2, though it would work equally well if manually installed without it.

[More information on making anaconda environments found here.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda create --name tpot2env python=3.10
conda activate tpot2env
```

### Note for M1 Mac or other Arm-based CPU users

You need to install the lightgbm package directly from conda using the following command before installing TPOT2. 

This is to ensure that you get the version that is compatible with your system.

```
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
```

### Developer/Latest Branch Installation


```
pip install -e /path/to/tpot2repo
```

If you downloaded with git pull, then the repository folder will be named TPOT2. (Note: this folder is the one that includes setup.py inside of it and not the folder of the same name inside it).
If you downloaded as a zip, the folder may be called tpot2-main. 


### Usage 

See the Tutorials Folder for instructions and examples.

## Contributing to TPOT2

We welcome you to check the existing issues for bugs or enhancements to work on. If you have an idea for an extension to TPOT, please file a new issue so we can discuss it.


### Known issues
* TPOT2 uses the func_timeout package to terminate long running pipelines. The early termination signal may fail on particular estimators and cause TPOT2 to run for longer than intended. If you are using your own custom configuration dictionaries, and are noticing that TPOT2 is running for longer than intended, this may be the issue. We are currently looking into it. Sometimes restarting TPOT2 resolves the issue.
* Periodic checkpoint folder may not correctly resume if using budget and/or initial_population size.
* Population class is slow to add new individuals. The Population class needs to be updated to use a dictionary for storage rather than a pandas dataframe.
* Crossover may sometimes go over the size restrictions.
* Memory caching with GraphPipeline may miss some nodes where the ordering on inputs happens to be different between two nodes. 




### Support for TPOT2

TPOT2 was developed in the [Artificial Intelligence Innovation (A2I) Lab](http://epistasis.org/) at Cedars-Sinai with funding from the [NIH](http://www.nih.gov/) under grants U01 AG066833 and R01 LM010098. We are incredibly grateful for the support of the NIH and the Cedars-Sinai during the development of this project.

The TPOT logo was designed by Todd Newmuis, who generously donated his time to the project.

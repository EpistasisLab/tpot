# Installation

TPOT requires a working installation of Python.

### Creating a conda environment (optional)

We recommend using conda environments for installing TPOT, though it would work equally well if manually installed without it.

[More information on making anaconda environments found here.](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

```
conda create --name tpotenv python=3.13
conda activate tpotenv
```

### Note for M1 Mac or other Arm-based CPU users

You need to install the lightgbm package directly from conda using the following command before installing TPOT. 

This is to ensure that you get the version that is compatible with your system.

```
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
```

### Developer/Latest Branch Installation


```
pip install -e /path/to/tpotrepo
```

If you downloaded with git pull, then the repository folder will be named TPOT. (Note: this folder is the one that includes setup.py inside of it and not the folder of the same name inside it).
If you downloaded as a zip, the folder may be called tpot-main. 

#!/bin/bash

# modified from https://github.com/trevorstephens/gplearn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.


# License: GNU/GPLv3

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
# deactivate

# Use the miniconda installer for faster download / install of conda
# itself

chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions

conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
    numpy scipy scikit-learn cython pandas joblib

source activate testenv

pip install deap tqdm update_checker stopit \
    dask[delayed] dask-ml xgboost cloudpickle==0.5.6

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import joblib; print('joblib %s' % joblib.__version__)"
python -c "import deap; print('deap %s' % deap.__version__)"
python -c "import xgboost; print('xgboost %s ' % xgboost.__version__)"
python -c "import update_checker; print('update_checker %s' % update_checker.__version__)"
python -c "import tqdm; print('tqdm %s' % tqdm.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import stopit; print('stopit %s' % stopit.__version__)"
python setup.py build_ext --inplace

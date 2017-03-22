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
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
wget http://repo.continuum.io/miniconda/Miniconda-3.9.1-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions
if [[ "$LATEST" == "true" ]]; then
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy scipy scikit-learn cython
else
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
        scikit-learn=$SKLEARN_VERSION \
    cython
fi

source activate testenv

if [[ "$LATEST" == "true" ]]; then
    pip install deap
    pip install xgboost
else
    pip install deap==$DEAP_VERSION
    pip install xgboost==$XGBOOST_VERSION
fi

pip install update_checker
pip install tqdm
pip install pathos

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

# build output in the travis output when it succeeds.
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import deap; print('deap %s' % deap.__version__)"
python -c "import xgboost; print('xgboost %s ' % xgboost.__version__)"
python -c "import update_checker; print('update_checker %s' % update_checker.__version__)"
python -c "import tqdm; print('tqdm %s' % tqdm.__version__)"
python -c "import pathos; print('pathos %s' % pathos.__version__)"
python setup.py build_ext --inplace

# modified from https://github.com/trevorstephens/gplearn

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: GNU/GPLv3

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import deap; print('deap %s' % deap.__version__)"
python -c "import xgboost; print('xgboost %s ' % xgboost.__version__)"
python -c "import update_checker; print('update_checker %s ' % update_checker.__version__)"
python -c "import tqdm; print('tqdm %s' % tqdm.__version__)"

if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage
else
    nosetests -s -v
fi
# make test-doc test-sphinxext

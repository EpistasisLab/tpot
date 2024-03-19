#TODO: how to best support transformers/selectors that take other transformers with their own hyperparameters?
import numpy as np
import sklearn

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

SelectFwe_configspace = ConfigurationSpace(
    space = {
        'alpha': Float('alpha', bounds=(1e-4, 0.05), log=True),
    }
)


SelectPercentile_configspace = ConfigurationSpace(
    space = {
        'percentile': Float('percentile', bounds=(1, 100.0)),
    }
)

VarianceThreshold_configspace = ConfigurationSpace(
    space = {
        'threshold': Float('threshold', bounds=(1e-4, .2), log=True),
    }
)



# Note the RFE_configspace_part and SelectFromModel_configspace_part are not complete, they both require the estimator to be set. 
# These are indended to be used with the Wrapped search space.
RFE_configspace_part = ConfigurationSpace(
    space = {
        'step': Float('step', bounds=(1e-4, 1.0)),
    }
)

SelectFromModel_configspace_part = ConfigurationSpace(
    space = {
        'threshold': Float('threshold', bounds=(1e-4, 1.0), log=True),
    }
)

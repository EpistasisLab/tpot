"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
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

"""
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

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
from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from ConfigSpace import EqualsCondition, OrConjunction, NotEqualsCondition, InCondition
import numpy as np

Binarizer_configspace = ConfigurationSpace(
    space = {
        'threshold': Float('threshold', bounds=(0.0, 1.0)),
    }
)

Normalizer_configspace = ConfigurationSpace(
    space={'norm': Categorical('norm', ['l1', 'l2', 'max'])}
)

PCA_configspace = ConfigurationSpace(
    space={'n_components': Float('n_components', bounds=(0.5, 0.999))}
)

ZeroCount_configspace = {}

PolynomialFeatures_configspace = ConfigurationSpace(
    space = {
        'degree': Integer('degree', bounds=(2, 3)),
        'interaction_only': Categorical('interaction_only', [True, False]),
    }
)

OneHotEncoder_configspace = {} #TODO include the parameter for max unique values

OrdinalEncoder_configspace = {} #TODO include the parameter for max unique values

def get_FastICA_configspace(n_features=100, random_state=None):
    
    space = {
        'n_components': Integer('n_components', bounds=(1, n_features)),
        'algorithm': Categorical('algorithm', ['parallel', 'deflation']),
        'whiten':'unit-variance',
    }
            
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space

    )

def get_FeatureAgglomeration_configspace(n_features):

    linkage = Categorical('linkage', ['ward', 'complete', 'average'])
    metric = Categorical('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'])
    n_clusters = Integer('n_clusters', bounds=(2, min(n_features,400)))
    pooling_func = Categorical('pooling_func', ['mean', 'median', 'max'])

    metric_condition = NotEqualsCondition(metric, linkage, 'ward')

    cs =  ConfigurationSpace()
    cs.add([linkage, metric, n_clusters, pooling_func])
    cs.add(metric_condition)
    
    return cs


def FeatureAgglomeration_hyperparameter_parser(params):
    new_params = params.copy()
    if "pooling_func" in new_params:
        if new_params["pooling_func"] == "mean":
            new_params["pooling_func"] = np.mean
        elif new_params["pooling_func"] == "median":
            new_params["pooling_func"] = np.median
        elif new_params["pooling_func"] == "max":
            new_params["pooling_func"] = np.max
        elif new_params["pooling_func"] == "min":
            new_params["pooling_func"] = np.min

    return new_params


def get_Nystroem_configspace(n_features=100, random_state=None,):

    space = {
        'gamma': Float('gamma', bounds=(0.0, 1.0)),
        'kernel': Categorical('kernel', ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid']),
        'n_components': Integer('n_components', bounds=(1, n_features)),
    }


    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space

    )

def get_RBFSampler_configspace(n_features=100, random_state=None):

    space = {
        'gamma': Float('gamma', bounds=(0.0, 1.0)),
        'n_components': Integer('n_components', bounds=(1, n_features)),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space

    )


def get_QuantileTransformer_configspace(random_state=None, n_samples=1000):

    space = {
        'n_quantiles': Integer('n_quantiles', bounds=(10, n_samples)),
        'output_distribution': Categorical('output_distribution', ['uniform', 'normal']),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space

    )


def get_passkbinsdiscretizer_configspace(random_state=None):
    space = {
        'n_bins': Integer('n_bins', bounds=(3, 100)),
        'encode': 'onehot-dense',
        'strategy': Categorical('strategy', ['uniform', 'quantile', 'kmeans']),
        # 'subsample': Categorical('subsample', ['auto', 'warn', 'ignore']),
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space

    )


### ROBUST SCALER

RobustScaler_configspace = ConfigurationSpace({
            "q_min": Float("q_min", bounds=(0.001, 0.3)),
            "q_max": Float("q_max", bounds=(0.7, 0.999)),
        })

def robust_scaler_hyperparameter_parser(params):
    return {"quantile_range": (params["q_min"], params["q_max"])}




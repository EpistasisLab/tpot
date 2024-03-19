from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal


Binarizer_configspace = ConfigurationSpace(
    space = {
        'threshold': Float('threshold', bounds=(0.0, 1.0)),
    }
)

Normalizer_configspace = ConfigurationSpace(
    space={'norm': Categorical('norm', ['l1', 'l2', 'max'])}
)

PCA_configspace = ConfigurationSpace(
    space={'n_components': Float('n_components', bounds=(0.001, 0.999))}
)

ZeroCount_configspace = ConfigurationSpace()

OneHotEncoder_configspace = ConfigurationSpace() #TODO include the parameter for max unique values

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

def get_FeatureAgglomeration_configspace(n_features=100):
    return ConfigurationSpace(
        space = {
            'linkage': Categorical('linkage', ['ward', 'complete', 'average']),
            'metric': Categorical('metric', ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']),
            'n_clusters': Integer('n_clusters', bounds=(2, n_features-1)),
        }
    )

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

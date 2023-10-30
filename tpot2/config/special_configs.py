from tpot2.builtin_modules import ArithmeticTransformer, FeatureSetSelector
from functools import partial
import pandas as pd
import numpy as np
from tpot2.builtin_modules import AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer

# ArithmeticTransformer
def params_arthmetic_operator(trial, name=None):
        return {
                'function': trial.suggest_categorical(f'function_{name}', ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]),
                }

def make_arithmetic_transformer_config_dictionary():
        return  {
                AddTransformer: {},
                mul_neg_1_Transformer: {},
                MulTransformer: {},
                SafeReciprocalTransformer: {},
                EQTransformer: {},
                NETransformer: {},
                GETransformer: {},
                GTTransformer: {},
                LETransformer: {},
                LTTransformer: {},
                MinTransformer: {},
                MaxTransformer: {},
        }





def params_feature_set_selector(trial, name=None, names_list = None, subset_dict=None):
    """Create a dictionary of parameters for FeatureSetSelector.

    Parameters
    ----------
    trial: optuna.trial.Trial
        A trial corresponds to the evaluation of a objective function.
    name: string
        Used for compatibility in when calling multiple optuna of multiple parameters at once.
    names_list: list of string
        List of names of the feature set selector. To more easily keep track of what the subsets represent.
        Included to prevent repeat calls to list(subset_dict.keys()) which may be slow and/or have different orderings
    subset_dict: dictionary
        A dictionary of subsets. The keys are the names of the subsets and the values are the subsets.

    Returns
    -------
    params: dictionary
        A dictionary of parameters for FeatureSetSelector.
    """

    subset_name = trial.suggest_categorical(f'subset_name_{name}', names_list)

    params =    {'name': subset_name,
                    'sel_subset': subset_dict[subset_name],
                }

    return params

def make_FSS_config_dictionary(subsets=None, n_features=None, feature_names=None):
    """Create the search space of parameters for FeatureSetSelector.

    Parameters
    ----------
    subsets: Sets the subsets to select from.
        - str : If a string, it is assumed to be a path to a csv file with the subsets.
            The first column is assumed to be the name of the subset and the remaining columns are the features in the subset.
        - list or np.ndarray : If a list or np.ndarray, it is assumed to be a list of subsets.

    n_features: int the number of features in the dataset.
        If subsets is None, each column will be treated as a subset. One column will be selected per subset.
    """

    #require at least of of the parameters
    if subsets is None and n_features is None:
        raise ValueError('At least one of the parameters must be provided')

    if isinstance(subsets, str):
        df = pd.read_csv(subsets,header=None,index_col=0)
        df['features'] = df.apply(lambda x: list([x[c] for c in df.columns]),axis=1)
        subset_dict = {}
        for row in df.index:
            subset_dict[row] = df.loc[row]['features']
    elif isinstance(subsets, dict):
        subset_dict = subsets
    elif isinstance(subsets, list) or isinstance(subsets, np.ndarray):
        subset_dict = {str(i):subsets[i] for i in range(len(subsets))}
    else:
        if feature_names is None:
            subset_dict = {str(i):i for i in range(n_features)}
        else:
            subset_dict = {str(i):feature_names[i] for i in range(len(feature_names))}

    names_list = list(subset_dict.keys())

    return {FeatureSetSelector: partial(params_feature_set_selector, names_list = names_list, subset_dict=subset_dict)}



from tpot2.builtin_modules import Passthrough

def params_passthrough(trial, name=None):
    return {}

def make_passthrough_config_dictionary():
    return {Passthrough: params_passthrough}
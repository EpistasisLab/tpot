from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
from ConfigSpace import EqualsCondition, OrConjunction, NotEqualsCondition, InCondition
from ..search_spaces.nodes.estimator_node import NONE_SPECIAL_STRING, TRUE_SPECIAL_STRING, FALSE_SPECIAL_STRING
from ..search_spaces.nodes import EstimatorNode
from ..search_spaces.pipelines import WrapperPipeline, ChoicePipeline, GraphPipeline
import ConfigSpace
import sklearn
from functools import partial
import inspect
import numpy as np

def load_get_module_from_string(module_string):
    module_name, class_name = module_string.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def hyperparameter_parser(hdict, function_params_conversion_dict):
    d = hdict.copy()
    d.update(function_params_conversion_dict)
    return d



def get_node_space(module_string, params):
    method = load_get_module_from_string(module_string)
    config_space = ConfigurationSpace()
    sub_space = None
    sub_space_name = None

    function_params_conversion_dict = {}

    if params is None:
        return EstimatorNode(method=method, space=config_space)

    for param_name, param in params.items():
        if param is None:
            config_space.add_hyperparameter(Categorical(param_name, [NONE_SPECIAL_STRING]))

        if isinstance(param, range):
            param = list(param)

        if isinstance(param, list) or isinstance(param, np.ndarray):
            if len(param) == 0:
                p = param[0]
                if p is None:
                    p = NONE_SPECIAL_STRING
                elif type(p) == bool:
                    p = TRUE_SPECIAL_STRING if p else FALSE_SPECIAL_STRING
                
                config_space.add_hyperparameter(ConfigSpace.hyperparameters.Constant(param_name, p))
            else:
                config_space.add_hyperparameter(Categorical(param_name, param))
            # if all(isinstance(i, int) for i in param):
            #     config_space.add_hyperparameter(Integer(param_name, (min(param), max(param))))
            # elif all(isinstance(i, float) for i in param):
            #     config_space.add_hyperparameter(Float(param_name, (min(param), max(param))))
            # else:
            #     config_space.add_hyperparameter(Categorical(param_name, param))
        elif isinstance(param, dict): #TPOT1 config dicts have dictionaries for values of hyperparameters that are either a function or an estimator
            if len(param) > 1:
                    raise ValueError(f"Multiple items in dictionary entry for {param_name}")
            
            key = list(param.keys())[0]

            innermethod = load_get_module_from_string(key)
            
            if inspect.isclass(innermethod) and issubclass(innermethod, sklearn.base.BaseEstimator): #is an estimator
                if sub_space is None:
                    sub_space_name = param_name
                    sub_space = get_node_space(key, param[key])   
                else:
                    raise ValueError("Only multiple hyperparameters are estimators. Only one parameter ")
        
            else: #assume the key is a function and ignore the value
                function_params_conversion_dict[param_name] = innermethod
        
        else:
            # config_space.add_hyperparameter(Categorical(param_name, param))
            config_space.add_hyperparameter(ConfigSpace.hyperparameters.Constant(param_name, param))

    parser=None
    if len(function_params_conversion_dict) > 0:
        parser = partial(hyperparameter_parser, function_params_conversion_dict)


    if sub_space is None:
    
        if parser is not None:
            return EstimatorNode(method=method, space=config_space, hyperparameter_parser=parser)
        else:
            return EstimatorNode(method=method, space=config_space)
    
    
    else:
        if parser is not None:
            return WrapperPipeline(method=method, space=config_space, estimator_search_space=sub_space, wrapped_param_name=sub_space_name, hyperparameter_parser=parser)
        else:
            return WrapperPipeline(method=method, space=config_space, estimator_search_space=sub_space, wrapped_param_name=sub_space_name)


def convert_config_dict_to_list(config_dict):
    search_spaces = []
    for key, value in config_dict.items():
        search_spaces.append(get_node_space(key, value))
    return search_spaces


def convert_config_dict_to_choicepipeline(config_dict):
    search_spaces = []
    for key, value in config_dict.items():
        search_spaces.append(get_node_space(key, value))
    return ChoicePipeline(search_spaces)

#Note doesn't convert estimators so they passthrough inputs like in TPOT1
def convert_config_dict_to_graphpipeline(config_dict):
    root_search_spaces = []
    inner_search_spaces = []

    for key, value in config_dict.items():
        #if root
        if issubclass(load_get_module_from_string(key), sklearn.base.ClassifierMixin) or issubclass(load_get_module_from_string(key), sklearn.base.RegressorMixin):
            root_search_spaces.append(get_node_space(key, value))
        else:
            inner_search_spaces.append(get_node_space(key, value))
        
    if len(root_search_spaces) == 0:
        Warning("No classifiers or regressors found, allowing any estimator to be the root node")
        root_search_spaces = inner_search_spaces

    #merge inner and root search spaces

    inner_space = np.concatenate([root_search_spaces,inner_search_spaces])

    root_space = ChoicePipeline(root_search_spaces)
    inner_space = ChoicePipeline(inner_search_spaces)

    final_space = GraphPipeline(root_search_space=root_space, inner_search_space=inner_space)
    return final_space
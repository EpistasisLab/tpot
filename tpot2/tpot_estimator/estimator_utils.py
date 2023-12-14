import numpy as np
import sklearn
import sklearn.base
import tpot2
import pandas as pd

from .cross_val_utils import cross_val_score_objective

def convert_parents_tuples_to_integers(row, object_to_int):
    if type(row) == list or type(row) == np.ndarray or type(row) == tuple:
        return tuple(object_to_int[obj] for obj in row)
    else:
        return np.nan

def apply_make_pipeline(graphindividual, preprocessing_pipeline=None):
    try:
        if preprocessing_pipeline is None:
            return graphindividual.export_pipeline()
        else:
            return sklearn.pipeline.make_pipeline(sklearn.base.clone(preprocessing_pipeline), graphindividual.export_pipeline())
    except:
        return None

def get_configuration_dictionary(options, n_samples, n_features, classification, random_state=None, cv=None, subsets=None, feature_names=None, n_classes=None):
    if options is None:
        return options

    if isinstance(options, dict):
        return recursive_with_defaults(options, n_samples, n_features, classification, random_state=None, cv=None, subsets=subsets, feature_names=feature_names, n_classes=n_classes)

    if not isinstance(options, list):
        options = [options]

    config_dict = {}

    for option in options:

        if option == "selectors":
            config_dict.update(tpot2.config.make_selector_config_dictionary(random_state=random_state, classifier=classification))

        elif option == "classifiers":
            config_dict.update(tpot2.config.make_classifier_config_dictionary(random_state=random_state, n_samples=n_samples, n_classes=n_classes))

        elif option == "classifiers_sklearnex":
            config_dict.update(tpot2.config.make_sklearnex_classifier_config_dictionary(random_state=random_state, n_samples=n_samples, n_classes=n_classes))

        elif option == "regressors":
            config_dict.update(tpot2.config.make_regressor_config_dictionary(random_state=random_state, cv=cv, n_samples=n_samples))

        elif option == "regressors_sklearnex":
            config_dict.update(tpot2.config.make_sklearnex_regressor_config_dictionary(random_state=random_state, n_samples=n_samples))

        elif option == "transformers":
            config_dict.update(tpot2.config.make_transformer_config_dictionary(random_state=random_state, n_features=n_features))

        elif option == "arithmetic_transformer":
            config_dict.update(tpot2.config.make_arithmetic_transformer_config_dictionary())

        elif option == "feature_set_selector":
            config_dict.update(tpot2.config.make_FSS_config_dictionary(subsets, n_features, feature_names=feature_names))

        elif option == "skrebate":
            config_dict.update(tpot2.config.make_skrebate_config_dictionary(n_features=n_features))

        elif option == "MDR":
            config_dict.update(tpot2.config.make_MDR_config_dictionary())

        elif option == "continuousMDR":
            config_dict.update(tpot2.config.make_ContinuousMDR_config_dictionary())

        elif option == "FeatureEncodingFrequencySelector":
            config_dict.update(tpot2.config.make_FeatureEncodingFrequencySelector_config_dictionary())

        elif option == "genetic encoders":
            config_dict.update(tpot2.config.make_genetic_encoders_config_dictionary())

        elif option == "passthrough":
            config_dict.update(tpot2.config.make_passthrough_config_dictionary())


        else:
            config_dict.update(recursive_with_defaults(option, n_samples, n_features, classification, random_state, cv, subsets=subsets, feature_names=feature_names, n_classes=n_classes))

    if len(config_dict) == 0:
        raise ValueError("No valid configuration options were provided. Please check the options you provided and try again.")

    return config_dict

def recursive_with_defaults(config_dict, n_samples, n_features, classification, random_state=None, cv=None, subsets=None, feature_names=None, n_classes=None):

    for key in 'leaf_config_dict', 'root_config_dict', 'inner_config_dict', 'Recursive':
        if key in config_dict:
            value = config_dict[key]
            if key=="Resursive":
                config_dict[key] = recursive_with_defaults(value, n_samples, n_features, classification, random_state, cv, subsets=None, feature_names=None, n_classes=None)
            else:
                config_dict[key] = get_configuration_dictionary(value, n_samples, n_features, classification, random_state, cv, subsets, feature_names, n_classes)

    return config_dict



def objective_function_generator(pipeline, x,y, scorers, cv, other_objective_functions, memory=None, cross_val_predict_cv=None, subset_column=None, step=None, budget=None, generation=1,is_classification=True):
    pipeline = pipeline.export_pipeline(memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column)
    if budget is not None and budget < 1:
        if is_classification:
            x,y = sklearn.utils.resample(x,y, stratify=y, n_samples=int(budget*len(x)), replace=False, random_state=1)
        else:
            x,y = sklearn.utils.resample(x,y, n_samples=int(budget*len(x)), replace=False, random_state=1)

        if isinstance(cv, int) or isinstance(cv, float):
            n_splits = cv
        else:
            n_splits = cv.n_splits

    if len(scorers) > 0:
        cv_obj_scores = cross_val_score_objective(sklearn.base.clone(pipeline),x,y,scorers=scorers, cv=cv , fold=step)
    else:
        cv_obj_scores = []

    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]
        #flatten
        other_scores = np.array(other_scores).flatten().tolist()
    else:
        other_scores = []

    return np.concatenate([cv_obj_scores,other_scores])

def val_objective_function_generator(pipeline, X_train, y_train, X_test, y_test, scorers, other_objective_functions, memory, cross_val_predict_cv, subset_column):
    #subsample the data
    pipeline = pipeline.export_pipeline(memory=memory, cross_val_predict_cv=cross_val_predict_cv, subset_column=subset_column)
    fitted_pipeline = sklearn.base.clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)

    if len(scorers) > 0:
        scores =[sklearn.metrics.get_scorer(scorer)(fitted_pipeline, X_test, y_test) for scorer in scorers]

    other_scores = []
    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]

    return np.concatenate([scores,other_scores])


def remove_underrepresented_classes(x, y, min_count):
    if isinstance(y, (np.ndarray, pd.Series)):
        unique, counts = np.unique(y, return_counts=True)
        if min(counts) >= min_count:
            return x, y
        keep_classes = unique[counts >= min_count]
        mask = np.isin(y, keep_classes)
        x = x[mask]
        y = y[mask]
    elif isinstance(y, pd.DataFrame):
        counts = y.apply(pd.Series.value_counts)
        if min(counts) >= min_count:
            return x, y
        keep_classes = counts.index[counts >= min_count].tolist()
        mask = y.isin(keep_classes).all(axis=1)
        x = x[mask]
        y = y[mask]
    else:
        raise TypeError("y must be a numpy array or a pandas Series/DataFrame")
    return x, y


def convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        return x




def check_if_y_is_encoded(y):
    '''
    checks if the target y is composed of sequential ints from 0 to N
    '''
    y = sorted(set(y))
    return all(i == j for i, j in enumerate(y))

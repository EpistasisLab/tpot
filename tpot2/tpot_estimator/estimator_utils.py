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

#TODO add kwargs
def apply_make_pipeline(graphindividual, preprocessing_pipeline=None, export_graphpipeline=False, **pipeline_kwargs):
    try:

        if export_graphpipeline:
            est = graphindividual.export_flattened_graphpipeline(**pipeline_kwargs)
        else:
            est = graphindividual.export_pipeline()


        if preprocessing_pipeline is None:
            return est
        else:
            return sklearn.pipeline.make_pipeline(sklearn.base.clone(preprocessing_pipeline), est)
    except:
        return None





def objective_function_generator(pipeline, x,y, scorers, cv, other_objective_functions, step=None, budget=None, generation=1, is_classification=True, export_graphpipeline=False, **pipeline_kwargs):
    #pipeline = pipeline.export_pipeline(**pipeline_kwargs)
    if export_graphpipeline:
        pipeline = pipeline.export_flattened_graphpipeline(**pipeline_kwargs)
    else:
        pipeline = pipeline.export_pipeline()

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

def val_objective_function_generator(pipeline, X_train, y_train, X_test, y_test, scorers, other_objective_functions, export_graphpipeline=False, **pipeline_kwargs):
    #subsample the data
    if export_graphpipeline:
        pipeline = pipeline.export_flattened_graphpipeline(**pipeline_kwargs)
    else:
        pipeline = pipeline.export_pipeline()

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

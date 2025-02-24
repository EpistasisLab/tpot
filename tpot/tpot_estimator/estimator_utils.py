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
import numpy as np
import sklearn
import sklearn.base
import tpot
import pandas as pd

from .cross_val_utils import cross_val_score_objective

def convert_parents_tuples_to_integers(row, object_to_int):
    """
    Helper function to convert the parent rows into integers representing the index of the parent in the population.
    
    Original pandas dataframe using a custom index for the parents. This function converts the custom index to an integer index for easier manipulation by end users.

    Parameters
    ----------
    row: list, np.ndarray, tuple
        The row to convert.
    object_to_int: dict
        A dictionary mapping the object to an integer index.

    Returns 
    -------
    tuple
        The row with the custom index converted to an integer index.
    """
    if type(row) == list or type(row) == np.ndarray or type(row) == tuple:
        return tuple(object_to_int[obj] for obj in row)
    else:
        return np.nan

#TODO add kwargs
def apply_make_pipeline(ind, preprocessing_pipeline=None, export_graphpipeline=False, **pipeline_kwargs):
    """
    Helper function to create a column of sklearn pipelines from the tpot individual class.

    Parameters
    ----------
    ind: tpot.SklearnIndividual
        The individual to convert to a pipeline.
    preprocessing_pipeline: sklearn.pipeline.Pipeline, optional
        The preprocessing pipeline to include before the individual's pipeline.
    export_graphpipeline: bool, default=False
        Force the pipeline to be exported as a graph pipeline. Flattens all nested pipelines, FeatureUnions, and GraphPipelines into a single GraphPipeline.
    pipeline_kwargs: dict
        Keyword arguments to pass to the export_pipeline or export_flattened_graphpipeline method.
    
    Returns
    -------
    sklearn estimator
    """
    
    try:

        if export_graphpipeline:
            est = ind.export_flattened_graphpipeline(**pipeline_kwargs)
        else:
            est = ind.export_pipeline(**pipeline_kwargs)


        if preprocessing_pipeline is None:
            return est
        else:
            return sklearn.pipeline.make_pipeline(sklearn.base.clone(preprocessing_pipeline), est)
    except:
        return None





def objective_function_generator(pipeline, x,y, scorers, cv, other_objective_functions, step=None, budget=None, is_classification=True, export_graphpipeline=False, **pipeline_kwargs):
    """
    Uses cross validation to evaluate the pipeline using the scorers, and concatenates results with scores from standalone other objective functions.

    Parameters
    ----------
    pipeline: tpot.SklearnIndividual
        The individual to evaluate.
    x: np.ndarray
        The feature matrix.
    y: np.ndarray
        The target vector.
    scorers: list
        The scorers to use for cross validation. 
    cv: int, float, or sklearn cross-validator
        The cross-validator to use. For example, sklearn.model_selection.KFold or sklearn.model_selection.StratifiedKFold.
        If an int, will use sklearn.model_selection.KFold with n_splits=cv.
    other_objective_functions: list
        A list of standalone objective functions to evaluate the pipeline. With signature obj(pipeline) -> float. or obj(pipeline) -> np.ndarray
        These functions take in the unfitted estimator.
    step: int, optional
        The fold to return the scores for. If None, will return the mean of all the scores (per scorer). Default is None.
    budget: float, optional
        The budget to subsample the data. If None, will use the full dataset. Default is None.
        Will subsample budget*len(x) samples.
    is_classification: bool, default=True
        If True, will stratify the subsampling. Default is True.
    export_graphpipeline: bool, default=False
        Force the pipeline to be exported as a graph pipeline. Flattens all nested sklearn pipelines, FeatureUnions, and GraphPipelines into a single GraphPipeline.
    pipeline_kwargs: dict
        Keyword arguments to pass to the export_pipeline or export_flattened_graphpipeline method.

    Returns
    -------
    np.ndarray
        The concatenated scores for the pipeline. The first len(scorers) elements are the cross validation scores, and the remaining elements are the standalone objective functions.
        
    """

    if export_graphpipeline:
        pipeline = pipeline.export_flattened_graphpipeline(**pipeline_kwargs)
    else:
        pipeline = pipeline.export_pipeline(**pipeline_kwargs)

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
    """
    Trains a pipeline on a training set and evaluates it on a test set using the scorers and other objective functions.

    Parameters
    ----------

    pipeline: tpot.SklearnIndividual
        The individual to evaluate.
    X_train: np.ndarray
        The feature matrix of the training set.
    y_train: np.ndarray
        The target vector of the training set.
    X_test: np.ndarray
        The feature matrix of the test set.
    y_test: np.ndarray
        The target vector of the test set.
    scorers: list
        The scorers to use for cross validation.
    other_objective_functions: list
        A list of standalone objective functions to evaluate the pipeline. With signature obj(pipeline) -> float. or obj(pipeline) -> np.ndarray
        These functions take in the unfitted estimator.
    export_graphpipeline: bool, default=False
        Force the pipeline to be exported as a graph pipeline. Flattens all nested sklearn pipelines, FeatureUnions, and GraphPipelines into a single GraphPipeline.
    pipeline_kwargs: dict
        Keyword arguments to pass to the export_pipeline or export_flattened_graphpipeline method.

    Returns
    -------
    np.ndarray
        The concatenated scores for the pipeline. The first len(scorers) elements are the cross validation scores, and the remaining elements are the standalone objective functions.
        

    """
    
    #subsample the data
    if export_graphpipeline:
        pipeline = pipeline.export_flattened_graphpipeline(**pipeline_kwargs)
    else:
        pipeline = pipeline.export_pipeline(**pipeline_kwargs)

    fitted_pipeline = sklearn.base.clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)

    if len(scorers) > 0:
        scores =[sklearn.metrics.get_scorer(scorer)(fitted_pipeline, X_test, y_test) for scorer in scorers]

    other_scores = []
    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]

    return np.concatenate([scores,other_scores])


def remove_underrepresented_classes(x, y, min_count):
    """
    Helper function to remove classes with less than min_count samples from the dataset.

    Parameters
    ----------
    x: np.ndarray or pd.DataFrame
        The feature matrix.
    y: np.ndarray or pd.Series
        The target vector.
    min_count: int
        The minimum number of samples to keep a class.

    Returns
    -------
    np.ndarray, np.ndarray
        The feature matrix and target vector with rows from classes with less than min_count samples removed.
    """
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
    Checks if the target y is composed of sequential ints from 0 to N.
    XGBoost requires the target to be encoded in this way.

    Parameters
    ----------
    y: np.ndarray
        The target vector.

    Returns
    -------
    bool
        True if the target is encoded as sequential ints from 0 to N, False otherwise
    '''
    y = sorted(set(y))
    return all(i == j for i, j in enumerate(y))

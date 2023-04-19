import types
from abc import abstractmethod
import numpy as np
from joblib import Parallel, delayed
import traceback
from collections.abc import Iterable
import warnings
from stopit import threading_timeoutable, TimeoutException
from tpot2.parent_selectors import survival_select_NSGA2
import time
import dask
import stopit
from dask.diagnostics import ProgressBar
from tqdm.dask import TqdmCallback
from dask.distributed import progress

import func_timeout

def process_scores(scores, n):
    '''
    Purpose: This function processes a list of scores to ensure that each score list has the same length, n. If a score list is shorter than n, the function fills the list with either "TIMEOUT" or "INVALID" values.

    Parameters:

        scores: A list of score lists. Each score list represents a set of scores for a particular player or team. The score lists may have different lengths.
        n: An integer representing the desired length for each score list.

    Returns:

        The scores list, after processing.
    
    '''
    for i in range(len(scores)):
        if len(scores[i]) < n:
            if "TIMEOUT" in scores[i]:
                scores[i] = ["TIMEOUT" for j in range(n)]
            else:
                scores[i] = ["INVALID" for j in range(n)]
    return scores


def objective_nan_wrapper(  individual, 
                            objective_function,
                            verbose=0,
                            timeout=None,
                            **objective_kwargs):
    with warnings.catch_warnings(record=True) as w:  #catches all warnings in w so it can be supressed by verbose                
        try:
            
            if timeout is None:
                value = objective_function(individual, **objective_kwargs)
            else:
                value = func_timeout.func_timeout(timeout, objective_function, args=[individual], kwargs=objective_kwargs)
            
            if not isinstance(value, Iterable):
                value = [value]               

            if len(w) and verbose>=4:
                
                warnings.warn(w[0].message)
            return value
        except func_timeout.exceptions.FunctionTimedOut:
            if verbose >= 4:
                print(f'WARNING AN INDIVIDUAL TIMED OUT')
            return ["TIMEOUT"]
        except Exception as e:
            if verbose == 4:
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION \n {e}')
            if verbose >= 5:
                trace = traceback.format_exc()
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION \n {e} \n {trace}')
            return ["INVALID"]
        

def eval_objective_list(ind, objective_list, verbose=0,**objective_kwargs):

    scores = np.concatenate([objective_nan_wrapper(ind, obj, verbose,**objective_kwargs) for obj in objective_list ])
    return scores

def parallel_eval_objective_list(individual_list,
                                objective_list,
                                n_jobs = 1,
                                verbose=0,
                                timeout=None,
                                n_expected_columns=None,
                                client=None,
                                **objective_kwargs):

    #offspring_scores = Parallel(n_jobs=n_jobs)(delayed(eval_objective_list)(ind,  objective_list, verbose, timeout=timeout)  for ind in individual_list )
    
    # delayed_values = [dask.delayed(eval_objective_list)(ind,  objective_list, verbose, timeout=timeout,**objective_kwargs)  for ind in individual_list]
    # with TqdmCallback(desc="Evaluating Individuals", disable=verbose<2, leave=False):
    #     offspring_scores = list(dask.compute( *delayed_values,
    #                             num_workers=n_jobs))
    # del delayed_values
    if client is None:
        client = dask.distributed.get_client()
    futures = [client.submit(eval_objective_list, ind,  objective_list, verbose, timeout=timeout,**objective_kwargs)  for ind in individual_list]
    
    if verbose >= 2:
        dask.distributed.progress(futures, notebook=False)
    
    dask.distributed.wait(futures)
    
    offspring_scores = []
    # todo optimize this
    for future in futures:
        if not future.exception():
            offspring_scores.append(future.result())
        else:
            offspring_scores.append(["INVALID"])

    if n_expected_columns is not None:
        offspring_scores = process_scores(offspring_scores, n_expected_columns)
    return offspring_scores


###################
# Parallel optimization
#############

@threading_timeoutable(np.nan) #TODO timeout behavior
def optimize_objective(ind, objective, steps=5, verbose=0):
    
    with warnings.catch_warnings(record=True) as w:  #catches all warnings in w so it can be supressed by verbose                
        try:
            value = ind.optimize(objective, steps=steps)
            if not isinstance(value, Iterable):
                value = [value]               

            if len(w) and verbose>=2:
                warnings.warn(w[0].message)
            return value
        except Exception as e:
            if verbose >= 2:
                print('WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION')
                print(e)
                print()
            if verbose >= 3:
                print(traceback.format_exc())
                print()
            return [np.nan]



def parallel_optimize_objective(individual_list,
                                objective,
                                n_jobs = 1,
                                verbose=0,
                                steps=5,
                                timeout=None,
                                **objective_kwargs,  ):

    Parallel(n_jobs=n_jobs)(delayed(optimize_objective)(ind,  objective,  steps, verbose, timeout=timeout)  for ind in individual_list ) #TODO: parallelize







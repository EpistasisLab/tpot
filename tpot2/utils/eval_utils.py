import types
from abc import abstractmethod
import numpy as np
from joblib import Parallel, delayed
import traceback
from collections.abc import Iterable
import warnings
from stopit import threading_timeoutable, TimeoutException
from tpot2.selectors import survival_select_NSGA2
import time
import dask
import stopit
from dask.diagnostics import ProgressBar
from tqdm.dask import TqdmCallback
from dask.distributed import progress
import distributed
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
                print(f'WARNING AN INDIVIDUAL TIMED OUT: \n {individual} \n')
            return ["TIMEOUT"]
        except Exception as e:
            if verbose == 4:
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION \n {individual} \n {e} \n')
            if verbose >= 5:
                trace = traceback.format_exc()
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION \n {individual} \n {e} \n {trace}')
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
                                parallel_timeout=None,
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
    
    if verbose >= 6:
        dask.distributed.progress(futures, notebook=False)
    
    try: 
        if parallel_timeout is not None and np.isinf(parallel_timeout):
            parallel_timeout = None
        dask.distributed.wait(futures, timeout=parallel_timeout)
    except dask.distributed.TimeoutError:
        print("terminating parallel evaluation due to timeout")
        pass
    except dask.distributed.CancelledError:
        print("dask.distributed.CancelledError")
    
    offspring_scores = []
    # todo optimize this
    for individual, future in zip(individual_list, futures):
        if not future.done():
            future.cancel()
            offspring_scores.append(["TIMEOUT"])
            if verbose >= 4:
                print(f'WARNING AN INDIVIDUAL TIMED OUT (Fallback): \n {individual} \n')
        elif future.exception():
            offspring_scores.append(["INVALID"])
            if verbose == 4:
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION (Dask Future) \n {individual} \n {future.exception()} \n')
            if verbose >= 5:
                print(f'WARNING THIS INDIVIDUAL CAUSED AND EXCEPTION (Dask Future) \n {individual} \n {future.exception()} \n {future.traceback()}')
        elif future.cancelled():
            offspring_scores.append(["INVALID"])
            if verbose == 4:
                print(f'WARNING THIS INDIVIDUAL WAS CANCELED BY DASK (likely memory issue) \n {individual}')

        else:
            offspring_scores.append(future.result())
            

    if n_expected_columns is not None:
        offspring_scores = process_scores(offspring_scores, n_expected_columns)
    return offspring_scores


def parallel_eval_objective_list2(individual_list,
                                objective_list,
                                verbose=0,
                                max_eval_time_seconds=None,
                                n_expected_columns=None,
                                client=None,
                                scheduled_timeout_time=None,
                                **objective_kwargs):

    individual_stack = list(individual_list)
    max_queue_size = len(client.cluster.workers)
    submitted_futures = {}
    scores_dict = {}
    submitted_inds = set()
    global_timeout_triggered = False
    while len(submitted_futures) < max_queue_size and len(individual_stack)>0:
        individual = individual_stack.pop()
        future = client.submit(eval_objective_list, individual,  objective_list, verbose=verbose, timeout=max_eval_time_seconds,**objective_kwargs)
        
        submitted_futures[future] = {"individual": individual,
                                    "time": time.time(),}
        
        submitted_inds.add(individual.unique_id())
    


    while len(individual_stack)>0 or len(submitted_futures)>0:
        #wait for at least one future to finish or timeout
        try:
            next(distributed.as_completed(submitted_futures, timeout=max_eval_time_seconds))
        except dask.distributed.TimeoutError:
            pass
        except dask.distributed.CancelledError:
            pass

        global_timeout_triggered = scheduled_timeout_time is not None and time.time() > scheduled_timeout_time

        #Loop through all futures, collect completed and timeout futures.
        for completed_future in list(submitted_futures.keys()):
            #get scores and update                            
            if completed_future.done(): #if future is done
                #If the future is done but threw and error, record the error
                if completed_future.exception() or completed_future.status == "error": #if the future is done and threw an error
                    print("Exception in future")
                    print(completed_future.exception())
                    scores = [np.nan for _ in range(n_expected_columns)]
                    eval_error = "INVALID"
                elif completed_future.cancelled(): #if the future is done and was cancelled
                    print("Cancelled future (likely memory related)")
                    scores = [np.nan for _ in range(n_expected_columns)]
                    eval_error = "INVALID"
                else: #if the future is done and did not throw an error, get the scores
                    try:
                        scores = completed_future.result()
                        #check if scores contain "INVALID" or "TIMEOUT"
                        if "INVALID" in scores:
                            eval_error = "INVALID"
                            scores = [np.nan for _ in range(n_expected_columns)]
                        elif "TIMEOUT" in scores:
                            eval_error = "TIMEOUT"
                            scores = [np.nan for _ in range(n_expected_columns)]
                        else:
                            eval_error = None
                        
                    except Exception as e:
                        print("Exception in future, but not caught by dask")
                        print(e)
                        print(completed_future.exception())
                        print(completed_future)
                        print("status", completed_future.status)
                        print("done", completed_future.done())
                        print("cancelld ", completed_future.cancelled())
                        scores = [np.nan for _ in range(n_expected_columns)]
                        eval_error = "INVALID"
            else: #if future is not done
                
                

                #check if the future has been running for too long, cancel the future
                if max_eval_time_seconds is not None and time.time() - submitted_futures[completed_future]["time"] > max_eval_time_seconds*1.25:
                    completed_future.cancel()
                    
                    if verbose >= 4:
                        print(f'WARNING AN INDIVIDUAL TIMED OUT (Fallback): \n {submitted_futures[completed_future]} \n')
                    
                    scores = [np.nan for _ in range(n_expected_columns)]
                    eval_error = "TIMEOUT"
                elif global_timeout_triggered:
                    completed_future.cancel()
                    
                    if verbose >= 4:
                        print(f'WARNING AN INDIVIDUAL TIMED OUT (max_time_seconds): \n {submitted_futures[completed_future]} \n')
                    
                    scores = [np.nan for _ in range(n_expected_columns)]
                    eval_error = None

                else:
                    continue #otherwise, continue to next future
        
            #log scores
            cur_individual = submitted_futures[completed_future]["individual"]
            scores_dict[cur_individual] = {"scores": scores, 
                                        "start_time": submitted_futures[completed_future]["time"],
                                        "end_time": time.time(),
                                        "eval_error": eval_error,
                                        }
            

            #update submitted futures
            submitted_futures.pop(completed_future)
        
        #break if timeout
        if global_timeout_triggered:
            while len(individual_stack) > 0:
                individual = individual_stack.pop()
                scores_dict[individual] = {"scores": [np.nan for _ in range(n_expected_columns)], 
                                        "start_time": time.time(),
                                        "end_time": time.time(),
                                        "eval_error": None,
                                        }
            break

        #submit new futures
        while len(submitted_futures) < max_queue_size and len(individual_stack)>0:
            individual = individual_stack.pop()
            future = client.submit(eval_objective_list, individual,  objective_list, verbose=verbose, timeout=max_eval_time_seconds,**objective_kwargs)
            
            submitted_futures[future] = {"individual": individual,
                                        "time": time.time(),}
            
            submitted_inds.add(individual.unique_id())


    #collect remaining futures


    final_scores = [scores_dict[individual]["scores"] for individual in individual_list]
    final_start_times = [scores_dict[individual]["start_time"] for individual in individual_list]
    final_end_times = [scores_dict[individual]["end_time"] for individual in individual_list]
    final_eval_errors = [scores_dict[individual]["eval_error"] for individual in individual_list]
    final_scores = process_scores(final_scores, n_expected_columns)
    return final_scores, final_start_times, final_end_times, final_eval_errors


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







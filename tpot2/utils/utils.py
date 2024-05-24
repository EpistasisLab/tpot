import numpy as np
import scipy
import statistics
import tpot2
import pandas as pd


def get_thresholds(scores, start=0, end=1, scale=.5, n=10,):
       thresh = beta_interpolation(start=start, end=end, scale=scale, n=n)
       return [np.percentile(scores, t) for t in thresh]

def equalize_list(lst, n_steps):
    step_size = len(lst) / n_steps
    new_lst = []
    for i in range(n_steps):
        start_index = int(i * step_size)
        end_index = int((i+1) * step_size)
        if i == 0: # First segment
            step_lst = [lst[start_index]] * (end_index - start_index)
        elif i == n_steps-1: # Last segment
            step_lst = [lst[-1]] * (end_index - start_index)
        else: # Middle segment
            segment = lst[start_index:end_index]
            median_value = statistics.median(segment)
            step_lst = [median_value] * (end_index - start_index)
        new_lst.extend(step_lst)
    return new_lst

def beta_interpolation(start=0, end=1, scale=1, n=10, n_steps=None):
    if n_steps is None:
        n_steps = n
    if n_steps > n:
        n_steps = n
    if scale <= 0:
        scale = 0.0001
    if scale >= 1:
        scale = 0.9999

    alpha = 3 * scale
    beta = 3 - alpha
    x = np.linspace(0,1,n)
    values = scipy.special.betainc(alpha,beta,x)*(end-start)+start

    if n_steps is not None:
        return equalize_list(values, n_steps)
    else:
        return values

#thanks chat gtp
def remove_items(items, indexes_to_remove):
    items = items.copy()
    #if items is a numpy array, we need to convert to a list
    if type(items) == np.ndarray:
        items = items.tolist()
    for index in sorted(indexes_to_remove, reverse=True):
        del items[index]
    return np.array(items)



# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# bigger is better
def is_pareto_efficient(scores, return_mask = True):
    """
    Find the pareto-efficient points
    :param scores: An (n_points, n_scores) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(scores.shape[0])
    n_points = scores.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(scores):
        nondominated_point_mask = np.any(scores>scores[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        scores = scores[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def get_pareto_frontier(df, column_names, weights):
    # dftmp = df[~df[column_names].isin(invalid_values).any(axis=1)]
    dftmp = df[df[column_names].notnull().all(axis=1)]

    if "Budget" in dftmp.columns:
        #get rows with the max budget
        dftmp = dftmp[dftmp["Budget"]==dftmp["Budget"].max()]


    indexes = dftmp[~dftmp[column_names].isna().any(axis=1)].index.values
    weighted_scores = df.loc[indexes][column_names].to_numpy()  * weights
    mask = is_pareto_efficient(weighted_scores, return_mask = True)
    df["Pareto_Front"] = np.nan #TODO this will get deprecated
    df.loc[indexes[mask], "Pareto_Front"] = 1
    


def get_pareto_front(df, column_names, weights):
    dftmp = df[df[column_names].notnull().all(axis=1)]

    if "Budget" in dftmp.columns:
        #get rows with the max budget
        dftmp = dftmp[dftmp["Budget"]==dftmp["Budget"].max()]


    indexes = dftmp[~dftmp[column_names].isna().any(axis=1)].index.values
    weighted_scores = df.loc[indexes][column_names].to_numpy()  * weights

    pareto_fronts = tpot2.selectors.nondominated_sorting(weighted_scores)

    df = pd.DataFrame(index=df.index,columns=["Pareto_Front"], data=[])
    
    df["Pareto_Front"] = np.nan

    for i, front in enumerate(pareto_fronts):
        for index in front:
            df.loc[indexes[index], "Pareto_Front"] = i+1

    return df["Pareto_Front"]
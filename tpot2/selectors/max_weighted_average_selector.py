import numpy as np

def max_weighted_average_selector(scores,k, n_parents=1, rng=None):
    """
    Select the best individual according to Max Weighted Average Selection, *k* times.

    Parameters
    ----------
    scores : np.ndarray
        The score matrix, where rows the individuals and the columns are the corresponds to scores on different objectives.
    k : int
        The number of individuals to select.
    n_parents : int, optional
        The number of parents to select per individual. The default is 1.
    rng : int, np.random.Generator, optional
        The random number generator. The default is None.
        
    Returns
    -------
        A array of indices of selected individuals of shape (k, n_parents).
        
    """
    ave_scores = [np.nanmean(s ) for s in scores ] #TODO make this more efficient
    chosen = np.argsort(ave_scores)[::-1][0:k] #TODO check this behavior with nans
    return np.reshape(chosen, (k, n_parents))
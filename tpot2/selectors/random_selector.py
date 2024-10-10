import numpy as np

def random_selector(scores,  k, n_parents=1, rng=None, ):
    """
    Randomly selects indeces of individuals from the scores matrix.

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
        A array of indices of randomly selected individuals (with replacement) of shape (k, n_parents).
    
    """
    rng = np.random.default_rng(rng)
    chosen = rng.choice(list(range(0,len(scores))), size=k*n_parents)
    return np.reshape(chosen, (k, n_parents))
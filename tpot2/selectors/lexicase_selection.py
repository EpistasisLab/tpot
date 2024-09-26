import numpy as np

def lexicase_selection(scores, k, n_parents=1, rng=None):
    """
    Select the best individual according to Lexicase Selection, *k* times.
    The returned list contains the indices of the chosen *individuals*.
    
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
    rng = np.random.default_rng(rng)
    chosen =[]
    for i in range(k*n_parents):
        candidates = list(range(len(scores)))
        cases = list(range(len(scores[0])))
        rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max(scores[candidates,cases[0]])
            candidates = [x for x in candidates if scores[x, cases[0]] == best_val_for_case]
            cases.pop(0)
        chosen.append(rng.choice(candidates))

    return np.reshape(chosen, (k, n_parents))
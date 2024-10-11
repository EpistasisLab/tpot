import numpy as np

def tournament_selection(scores, k, n_parents=1, rng=None, tournament_size=2, score_index=0):
    """
    Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The returned list contains the indices of the chosen *individuals*.
    
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
    tournament_size : int, optional
        The number of individuals participating in each tournament.
    score_index : int, str, optional
        The index of the score to use for selection. If "average" is passed, the average score is used. The default is 0 (only the first score is used).

    Returns
    -------
        A array of indices of selected individuals of shape (k, n_parents).
    """

    rng = np.random.default_rng(rng)

    if isinstance(score_index,int):
        key=lambda x:x[1][score_index]
    elif score_index == "average":
        key=lambda x:np.mean(x[1])

    chosen = []
    for i in range(k*n_parents):
        aspirants_idx =[rng.choice(len(scores)) for i in range(tournament_size)]
        aspirants  = list(zip(aspirants_idx, scores[aspirants_idx])) # Zip indices and elements together
        chosen.append(max(aspirants, key=key)[0]) # Retrun the index of the maximum element

    return np.reshape(chosen, (k, n_parents))
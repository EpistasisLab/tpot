import numpy as np

def tournament_selection(scores, k, rng=None, n_parents=1, tournament_size=2, score_index=0):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param score_index: The number of individuals participating in each tournament.
    :returns: A list of indices of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
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
import numpy as np

def lexicase_selection(scores, k, rng=None, n_parents=1,):
    """Select the best individual according to Lexicase Selection, *k* times.
    The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :returns: A list of indices of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
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
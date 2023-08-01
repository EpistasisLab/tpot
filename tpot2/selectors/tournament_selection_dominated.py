import numpy as np
import random

from.nsgaii import nondominated_sorting, crowding_distance, dominates

#based on deap
def tournament_selection_dominated(scores, k, n_parents=2):
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
    pareto_fronts = nondominated_sorting(scores)

    # chosen = list(itertools.chain.from_iterable(fronts))
    # if len(chosen) >= k:
    #     return chosen[0:k]

    crowding_dict = {}
    chosen = []
    current_front_number = 0
    while current_front_number < len(pareto_fronts):

        current_front = np.array(list(pareto_fronts[current_front_number]))
        front_scores = [scores[i] for i in current_front]
        crowding_distances = crowding_distance(front_scores)
        for i, crowding in zip(current_front,crowding_distances):
            crowding_dict[i] = crowding

        current_front_number += 1


    chosen = []
    for i in range(k*n_parents):
        asp1 = random.randrange(len(scores))
        asp2 = random.randrange(len(scores))

        if dominates(scores[asp1], scores[asp2]):
            chosen.append(asp1)
        elif dominates(scores[asp2], scores[asp1]):
            chosen.append(asp2)
        
        elif crowding_dict[asp1] > crowding_dict[asp2]:
            chosen.append(asp1)
        elif crowding_dict[asp1] < crowding_dict[asp2]:
            chosen.append(asp2)

        else:
            chosen.append(random.choice([asp1,asp2]))
    
    return np.reshape(chosen, (k, n_parents))







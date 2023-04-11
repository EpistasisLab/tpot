import numpy as np
import random


def random_selector(scores,  k, n_parents=1,): 
    chosen = random.choices(list(range(0,len(scores))), k=k*n_parents)
    return np.reshape(chosen, (k, n_parents))

def max_weighted_average_selector(scores,k, n_parents=1,): 
    ave_scores = [np.nanmean(s ) for s in scores ] #TODO make this more efficient
    chosen = np.argsort(ave_scores)[::-1][0:k] #TODO check this behavior with nans
    return np.reshape(chosen, (k, n_parents))

#based on deap
def TournamentSelection_Dominated(scores, k, n_parents=2):
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


def TournamentSelection(scores, k, n_parents=1, tournament_size=2, score_index=0):
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

    if isinstance(score_index,int):
        key=lambda x:x[1][score_index]
    elif score_index == "average":
        key=lambda x:np.mean(x[1])

    chosen = []
    for i in range(k*n_parents):
        aspirants_idx =[random.randrange(len(scores)) for i in range(tournament_size)]
        aspirants  = list(zip(aspirants_idx, scores[aspirants_idx])) # Zip indices and elements together
        chosen.append(max(aspirants, key=key)[0]) # Retrun the index of the maximum element
    
    return np.reshape(chosen, (k, n_parents))


def LexicaseSelection(scores, k, n_parents=1,):
    """Select the best individual according to Lexicase Selection, *k* times. 
    The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :returns: A list of indices of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen =[]
    for i in range(k*n_parents):
        candidates = list(range(len(scores)))
        cases = list(range(len(scores[0])))
        random.shuffle(cases)
        
        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max(scores[candidates,cases[0]])
            candidates = [x for x in candidates if scores[x, cases[0]] == best_val_for_case]
            cases.pop(0)
        chosen.append(random.choice(candidates))

    return np.reshape(chosen, (k, n_parents))


# Deb, Pratab, Agarwal, and Meyarivan, “A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II”, 2002.
# chatgpt

def nondominated_sorting(matrix):
    """
    Returns the indexes of the matrix 
    bigger is better
    """
    # Initialize the front list and the rank list

    # Initialize the current front
    fronts = {0:set()}

    # Initialize the list of dominated points
    dominated = [set() for _ in range(len(matrix))] #si the set of solutions which solution i dominates

    # Initialize the list of points that dominate the current point
    dominating = [0 for _ in range(len(matrix))] #ni the number of solutions that denominate solution i

    
    # Iterate over all points
    for p, p_scores in enumerate(matrix):
        # Iterate over all other points
        for q, q_scores in enumerate(matrix):
            # If the current point dominates the other point, increment the count of points dominated by the current point
            if dominates(p_scores, q_scores):
                dominated[p].add(q)
            # If the current point is dominated by the other point, add it to the list of dominated points
            elif dominates(q_scores, p_scores):
                dominating[p] += 1
        
        if dominating[p] == 0:
            fronts[0].add(p)

    i=0

    # Iterate until all points have been added to a front
    while len(fronts[i]) > 0:
        H = set()
        for p in fronts[i]:
            for q in dominated[p]:
                dominating[q] -= 1
                if dominating[q] == 0:
                    H.add(q)

        i += 1
        fronts[i] = H


    return [fronts[j] for j in range(i)]


def dominates(list1, list2):
    """
    returns true is all values in list1 are not strictly worse than list2 AND at least one item in list1 is better than list2
    """
    return all(list1[i] >= list2[i] for i in range(len(list1))) and any(list1[i] > list2[i] for i in range(len(list1)))

#adapted from deap + gtp
#bigger is better
def crowding_distance(matrix):
    matrix = np.array(matrix)
    # Initialize the crowding distance for each point to zero
    crowding_distances = [0 for _ in range(len(matrix))]
    
    # Iterate over each objective
    for objective_i in range(matrix.shape[1]):
        # Sort the points according to the current objective
        sorted_i = matrix[:, objective_i].argsort()
        
        # Set the crowding distance of the first and last points to infinity
        crowding_distances[sorted_i[0]] = float("inf")
        crowding_distances[sorted_i[-1]] = float("inf")
        
        if matrix[sorted_i[0]][objective_i] == matrix[sorted_i[-1]][objective_i]: # https://github.com/DEAP/deap/blob/f2a570567fa3dce156d7cfb0c50bc72f133258a1/deap/tools/emo.py#L135
            continue

        norm = matrix.shape[1] * float(matrix[sorted_i[0]][objective_i] - matrix[sorted_i[-1]][objective_i])
        for prev, cur, following in zip(sorted_i[:-2], sorted_i[1:-1], sorted_i[2:]):
            crowding_distances[cur] += (matrix[following][objective_i] - matrix[prev][objective_i]) / norm


    return crowding_distances




def survival_select_NSGA2(scores, k,):

    pareto_fronts = nondominated_sorting(scores)

    # chosen = list(itertools.chain.from_iterable(fronts))
    # if len(chosen) >= k:
    #     return chosen[0:k]

    chosen = []
    current_front_number = 0
    while len(chosen) < k and current_front_number < len(pareto_fronts):

        current_front = np.array(list(pareto_fronts[current_front_number]))
        front_scores = [scores[i] for i in current_front]
        crowding_distances = crowding_distance(front_scores)

        sorted_indeces = current_front[np.argsort(crowding_distances)[::-1]]

        chosen.extend(sorted_indeces[0:(k-len(chosen))])

        current_front_number += 1
    
    return chosen



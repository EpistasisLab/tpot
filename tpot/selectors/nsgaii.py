"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np

# Deb, Pratab, Agarwal, and Meyarivan, “A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization: NSGA-II”, 2002.
# chatgpt

def nondominated_sorting(matrix):
    """
    Returns the indices of the non-dominated rows in the scores matrix.
    Rows are considered samples, and columns are considered objectives.

    Parameters
    ----------
    matrix : np.ndarray
        The score matrix, where rows the individuals and the columns are the corresponds to scores on different objectives.

    Returns
    -------
    list
        A list of lists of indices of the non-dominated rows in the scores matrix.

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
    
    Parameters
    ----------
    list1 : list
        The first list of values to compare.
    list2 : list
        The second list of values to compare.
    
    Returns
    -------
    bool
        True if all values in list1 are not strictly worse than list2 AND at least one item in list1 is better than list2, False otherwise.
    
    """
    return all(list1[i] >= list2[i] for i in range(len(list1))) and any(list1[i] > list2[i] for i in range(len(list1)))

#adapted from deap + gtp
#bigger is better
def crowding_distance(matrix):
    """
    Takes a matrix of scores and returns the crowding distance for each point.

    Parameters
    ----------
    matrix : np.ndarray
        The score matrix, where rows the individuals and the columns are the corresponds to scores on different objectives.

    Returns
    -------
    list
        A list of the crowding distances for each point in the score matrix.
    """
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




def survival_select_NSGA2(scores, k, rng=None):
    """
    Select the top k individuals from the scores matrix using the NSGA-II algorithm.

    Parameters
    ----------
    scores : np.ndarray
        The score matrix, where rows the individuals and the columns are the corresponds to scores on different objectives.
    k : int
        The number of individuals to select.
    rng : int, np.random.Generator, optional
        The random number generator. The default is None.

    Returns
    -------
    list
        A list of indices of the selected individuals (without repeats).
    
    """

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
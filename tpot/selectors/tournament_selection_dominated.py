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

from.nsgaii import nondominated_sorting, crowding_distance, dominates

#based on deap
def tournament_selection_dominated(scores, k, n_parents=2, rng=None):
    """
    Select the best individual among 2 randomly chosen
    individuals, *k* times. Selection is first attempted by checking if one individual dominates the other. Otherwise one with the highest crowding distance is selected.
    The returned list contains the indices of the chosen *individuals*.
    
    Parameters
    ----------
    scores : np.ndarray
        The score matrix, where rows the individuals and the columns are the corresponds to scores on different objectives.
    k : int
        The number of individuals to select.
    n_parents : int, optional
        The number of parents to select per individual. The default is 2.
    rng : int, np.random.Generator, optional
        The random number generator. The default is None.
    
    Returns
    -------
        A array of indices of selected individuals of shape (k, n_parents).

    """

    rng = np.random.default_rng(rng)
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
        asp1 = rng.choice(len(scores))
        asp2 = rng.choice(len(scores))

        if dominates(scores[asp1], scores[asp2]):
            chosen.append(asp1)
        elif dominates(scores[asp2], scores[asp1]):
            chosen.append(asp2)

        elif crowding_dict[asp1] > crowding_dict[asp2]:
            chosen.append(asp1)
        elif crowding_dict[asp1] < crowding_dict[asp2]:
            chosen.append(asp2)

        else:
            chosen.append(rng.choice([asp1,asp2]))

    return np.reshape(chosen, (k, n_parents))

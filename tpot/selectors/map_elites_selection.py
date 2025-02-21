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
#TODO make these functions take in a predetermined set of bins rather than calculating a new set each time

def create_nd_matrix(matrix, grid_steps=None, bins=None):
    """
    Create an n-dimensional matrix with the highest score for each cell
    
    Parameters
    ----------
    matrix : np.ndarray
        The score matrix, where the first column is the score and the rest are the features for the map-elites algorithm.
    grid_steps : int, optional
        The number of steps to use for each feature to automatically create the bin thresholds. The default is None.
    bins : list, optional
        A list of lists containing the bin edges for each feature (other than the score). The default is None.

    Returns
    -------
    np.ndarray
        An n-dimensional matrix with the highest score for each cell and the index of the individual with that score.
        The value in the cell is a dictionary with the keys "score" and "idx" containing the score and index of the individual respectively.
    """
    if grid_steps is not None and bins is not None:
        raise ValueError("Either grid_steps or bins must be provided but not both")

    # Extract scores and features
    scores = matrix[:, 0]
    features = matrix[:, 1:]

    # Determine the min and max of each feature
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)

    # Create bins for each feature
    if bins is None:
        bins = [np.linspace(min_vals[i], max_vals[i], grid_steps) for i in range(len(min_vals))]

    # Initialize n-dimensional matrix with negative infinity
    nd_matrix = np.full([len(b)+1 for b in bins], {"score": -np.inf, "idx": None})
    # Fill in each cell with the highest score for that cell
    for idx, (score, feature) in enumerate(zip(scores, features)):
        indices = [np.digitize(f, bin) for f, bin in zip(feature, bins)]
        cur_score = nd_matrix[tuple(indices)]["score"]
        if score > cur_score:
            nd_matrix[tuple(indices)] = {"score": score, "idx": idx}

    return nd_matrix

def manhattan(a, b):
    """
    Calculate the Manhattan distance between two points.
    
    Parameters
    ----------
    a : np.ndarray
        The first point.
    b : np.ndarray
        The second point.

    Returns
    -------
    float
        The Manhattan distance between the two points.
    """
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def map_elites_survival_selector(scores,  k=None, rng=None, grid_steps= 10, bins=None):
    """
    Takes a matrix of scores and returns the indexes of the individuals that are in the best cells of the map-elites grid.
    Can either take a grid_steps parameter to automatically create the bins or a bins parameter to specify the bins manually.
    
    Parameters
    ----------
    scores : np.ndarray
        The score matrix, where the first column is the score and the rest are the features for the map-elites algorithm.
    k : int, optional
        The number of individuals to select. The default is None.
    rng : int, np.random.Generator, optional
        The random number generator. The default is None.
    grid_steps : int, optional
        The number of steps to use for each feature to automatically create the bin thresholds. The default is None.
    bins : list, optional
        A list of lists containing the bin edges for each feature (other than the score). The default is None.

    Returns
    -------
    np.ndarray
        An array of indexes of the individuals in the best cells of the map-elites grid (without repeats).
        
    """

    if grid_steps is not None and bins is not None:
        raise ValueError("Either grid_steps or bins must be provided but not both")

    rng = np.random.default_rng(rng)
    scores = np.array(scores)
    #create grid
    
    matrix = create_nd_matrix(scores, grid_steps=grid_steps, bins=bins)
    matrix = matrix.flatten()

    indexes =  [cell["idx"] for cell in matrix if cell["idx"] is not None]

    return np.unique(indexes)

def map_elites_parent_selector(scores,  k, n_parents=1, rng=None, manhattan_distance = 2,  grid_steps= 10, bins=None):
    """
    A parent selection algorithm for the map-elites algorithm. First creates a grid of the best individuals per cell and then selects parents based on the Manhattan distance between the cells of the best individuals.
    
    Parameters
    ----------
    scores : np.ndarray
        The score matrix, where the first column is the score and the rest are the features for the map-elites algorithm.
    k : int
        The number of individuals to select.
    n_parents : int, optional
        The number of parents to select per individual. The default is 1.
    rng : int, np.random.Generator, optional
        The random number generator. The default is None.
    manhattan_distance : int, optional
        The maximum Manhattan distance between parents. The default is 2. If no parents are found within this distance, the distance is increased by 1 until at least one other parent is found.
    grid_steps : int, optional
        The number of steps to use for each feature to automatically create the bin thresholds. The default is None.
    bins : list, optional
        A list of lists containing the bin edges for each feature (other than the score). The default is None.

    Returns
    -------
    np.ndarray
        An array of indexes of the parents selected for each individual

    """

    
    if grid_steps is not None and bins is not None:
        raise ValueError("Either grid_steps or bins must be provided but not both")
    
    rng = np.random.default_rng(rng)
    scores = np.array(scores)
    #create grid
    
    matrix = create_nd_matrix(scores, grid_steps=grid_steps, bins=bins)
    
    #return true if cell is not empty
    f = np.vectorize(lambda x: x["idx"] is not None)
    valid_coordinates  = np.array(np.where(f(matrix))).T

    idx_to_coordinates = {matrix[tuple(coordinates)]["idx"]: coordinates for coordinates in valid_coordinates}

    idxes = [idx for idx in idx_to_coordinates.keys()] #all the indexes of best score per cell

    distance_matrix = np.zeros((len(idxes), len(idxes)))

    for i, idx1 in enumerate(idxes):
        for j, idx2 in enumerate(idxes):
            distance_matrix[i][j] = manhattan(idx_to_coordinates[idx1], idx_to_coordinates[idx2])

    
    parents = []

    for i in range(k):
        #randomly select a cell
        idx = rng.choice(idxes) #select random parent

        #get the distance from this parent to all other parents 
        dm_idx = idxes.index(idx) 
        row = distance_matrix[dm_idx] 

        #get all second parents that are within manhattan distance. if none are found increase the distance
        candidates = []
        while len(candidates) == 0:
            candidates = np.where(row <= manhattan_distance)[0]
            #remove self from candidates
            candidates = candidates[candidates != dm_idx]
            manhattan_distance += 1

            if manhattan_distance > np.max(distance_matrix):
                break
        
        if len(candidates) == 0:
            parents.append([idx, idx]) #if no other parents are found, select the same parent twice. weird to crossover with itself though
        else:
            this_parents = [idx]
            for p in range(n_parents-1):
                idx2_cords = rng.choice(candidates)
                this_parents.append(idxes[idx2_cords])
            parents.append(this_parents)
        
    return np.array(parents)


def get_bins_quantiles(arr, k=None, q=None):
    """
    Takes a matrix and returns the bin thresholds based on quantiles.

    Parameters
    ----------
    arr : np.ndarray
        The matrix to calculate the bins for.
    k : int, optional
        The number of bins to create. This parameter creates k equally spaced quantiles. 
        For example, k=3 will create quantiles at array([0.25, 0.5 , 0.75]).
    q : np.ndarray, optional
        Custom quantiles to use for the bins. This parameter creates bins based on the quantiles of the data. The default is None.
    """
    bins = []

    if q is not None and k is not None:
        raise ValueError("Only one of k or q can be specified")

    if q is not None:
        final_q = q
    elif k is not None:
        final_q = np.linspace(0, 1, k+2)[1:-1]

    for i in range(arr.shape[1]):
        bins.append(np.quantile(arr[:,i], final_q))
    return bins

def get_bins(arr, k):
    """
    Get equally spaced bin thresholds between the min and max values for the array of scores.

    Parameters
    ----------
    arr : np.ndarray
        The list of values to calculate the bins for.
    k : int
        The number of bins to create.

    Returns
    -------
    list
        A list of bin thresholds calculated to be k equally spaced bins between the min and max of the array.
    
    """
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    [np.linspace(min_vals[i], max_vals[i], k) for i in range(len(min_vals))]
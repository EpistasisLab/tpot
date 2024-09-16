import numpy as np
#TODO make these functions take in a predetermined set of bins rather than calculating a new set each time

def create_nd_matrix(matrix, grid_steps=None, bins=None):

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
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def map_elites_survival_selector(scores,  k=None, rng=None, grid_steps= 10, bins=None):

    if grid_steps is not None and bins is not None:
        raise ValueError("Either grid_steps or bins must be provided but not both")

    rng = np.random.default_rng(rng)
    scores = np.array(scores)
    #create grid
    
    matrix = create_nd_matrix(scores, grid_steps=grid_steps, bins=bins)
    matrix = matrix.flatten()

    indexes =  [cell["idx"] for cell in matrix if cell["idx"] is not None]

    return np.unique(indexes)

def map_elites_parent_selector(scores,  k, rng=None, manhattan_distance = 2, n_parents=1, grid_steps= 10, bins=None):
    
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
    bins = []

    if q is not None and k is not None:
        raise ValueError("Only one of k or q can be specified")

    if q is not None:
        final_q = q
    elif k is not None:
        final_q = np.linspace(0, 1, k)

    for i in range(arr.shape[1]):
        bins.append(np.quantile(arr[:,i], final_q))
    return bins

def get_bins(arr, k):
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    [np.linspace(min_vals[i], max_vals[i], k) for i in range(len(min_vals))]
import numpy as np

def max_weighted_average_selector(scores,k, rng=None, n_parents=1,):
    ave_scores = [np.nanmean(s ) for s in scores ] #TODO make this more efficient
    chosen = np.argsort(ave_scores)[::-1][0:k] #TODO check this behavior with nans
    return np.reshape(chosen, (k, n_parents))
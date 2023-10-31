import numpy as np

def random_selector(scores,  k, rng_=None, n_parents=1, ):
    rng = np.random.default_rng(rng_)
    chosen = rng.choice(list(range(0,len(scores))), size=k*n_parents)
    return np.reshape(chosen, (k, n_parents))
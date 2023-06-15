import numpy as np
import random

def random_selector(scores,  k, n_parents=1,): 
    chosen = random.choices(list(range(0,len(scores))), k=k*n_parents)
    return np.reshape(chosen, (k, n_parents))
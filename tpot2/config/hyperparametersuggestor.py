import random
from scipy.stats import loguniform, logser #TODO: remove this dependency?
import numpy as np #TODO: remove this dependency and use scipy instead?




#Replicating the API found in optuna: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
#copy-pasted some code
def suggest_categorical(name, choices):
    return random.choice(choices)

def suggest_float(
    name: str,
    low: float,
    high: float,
    *,
    step = None,
    log = False,
    ):
    
    if log and step is not None:
        raise ValueError("The parameter `step` is not supported when `log` is true.")

    if low > high:
        raise ValueError(
            "The `low` value must be smaller than or equal to the `high` value "
            "(low={}, high={}).".format(low, high)
        )

    if log and low <= 0.0:
        raise ValueError(
            "The `low` value must be larger than 0 for a log distribution "
            "(low={}, high={}).".format(low, high)
        )

    if step is not None and step <= 0:
        raise ValueError(
            "The `step` value must be non-zero positive value, " "but step={}.".format(step)
        )

    #TODO check this produces correct output
    if log:
        value = np.random.uniform(np.log(low),np.log(high))
        return np.e**value

    else:
        if step is not None:
            return np.random.choice(np.arange(low,high,step))
        else:
            return np.random.uniform(low,high)


def suggest_discrete_uniform(name, low, high, q):
    return suggest_float(name, low, high, step=q)


def suggest_int(name, low, high, step=1, log=False):
    if low == high: #TODO check that this matches optuna's behaviour
        return low
    
    if log and step >1:
        raise ValueError("The parameter `step`>1 is not supported when `log` is true.")

    if low > high:
        raise ValueError(
            "The `low` value must be smaller than or equal to the `high` value "
            "(low={}, high={}).".format(low, high)
        )

    if log and low <= 0.0:
        raise ValueError(
            "The `low` value must be larger than 0 for a log distribution "
            "(low={}, high={}).".format(low, high)
        )

    if step is not None and step <= 0:
        raise ValueError(
            "The `step` value must be non-zero positive value, " "but step={}.".format(step)
        )

    if log:
        value = np.random.uniform(np.log(low),np.log(high))
        return int(np.e**value)
    else:
        return np.random.choice(list(range(low,high,step)))

def suggest_uniform(name, low, high):
    return suggest_float(name, low, high)
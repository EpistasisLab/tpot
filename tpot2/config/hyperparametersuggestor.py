# import random
# from scipy.stats import loguniform, logser #TODO: remove this dependency?
import numpy as np

#function that selects selects items from a list with each having independent probability p of being selected
def select(items, p, rng_=None):
    rng = np.random.default_rng(rng_)

    selected = [item for item in items if rng.random() < p]
    #if selected is empty, select one item at random
    if not selected:
        return [rng.choice(items)]
    return selected


class Trial():

    def __init__(self, rng_=None, old_params=None, alpha=1, hyperparameter_probability=1):
        self.rng = np.random.default_rng(rng_)

        self._params = dict()

        self.old_params = old_params
        self.alpha = alpha
        self.hyperparameter_probability = hyperparameter_probability

        if old_params is not None and len(old_params) > 0:
            self.params_to_update = select(list(old_params.keys()), self.hyperparameter_probability, rng_=self.rng)
        else:
            self.params_to_update = None


    #Replicating the API found in optuna: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    #copy-pasted some code
    def suggest_categorical(self, name, choices):
        if self.params_to_update ==  None or name in self.params_to_update or name not in self.old_params: #If this parameter is selected to be changed
            choice = self.suggest_categorical_(name, choices)
        else: #if this parameter is not selected to be changed
            choice = self.old_params[name]
            if choice not in choices: #if the old value is not in the choices, then we need to choose a value for it
                choice = self.suggest_categorical_(name, choices)

        self._params[name] = choice
        return choice

    def suggest_float(self,
                        name: str,
                        low: float,
                        high: float,
                        *,
                        step = None,
                        log = False,
                        ):
        if self.params_to_update ==  None or name in self.params_to_update or name not in self.old_params: #If this parameter is selected to be changed
            choice = self.suggest_float_(name, low=low, high=high, step=step, log=log)
            if self.old_params is not None and name in self.old_params:
                choice = self.alpha*choice + (1-self.alpha)*self.old_params[name]
        else: #if this parameter is not selected to be changed
                choice = self.old_params[name]

        self._params[name] = choice
        return choice



    def suggest_discrete_uniform(self, name, low, high, q):
        if self.params_to_update ==  None or name in self.params_to_update or name not in self.old_params:
            choice = self.suggest_discrete_uniform_(name, low=low, high=high, q=q)
            if self.old_params is not None and name in self.old_params:
                choice = self.alpha*choice + (1-self.alpha)*self.old_params[name]
        else:
            choice = self.old_params[name]

        self._params[name] = choice
        return choice



    def suggest_int(self, name, low, high, step=1, log=False):
        if self.params_to_update ==  None or name in self.params_to_update or name not in self.old_params:
            choice = self.suggest_int_(name, low=low, high=high, step=step, log=log)
            if self.old_params is not None and name in self.old_params:
                choice = int(self.alpha*choice + (1-self.alpha)*self.old_params[name])
        else:
            choice = self.old_params[name]

        self._params[name] = choice
        return choice


    def suggest_uniform(self, name, low, high):
        if self.params_to_update ==  None or name in self.params_to_update or name not in self.old_params:
            choice = self.suggest_uniform_(name, low=low, high=high)
            if self.old_params is not None and name in self.old_params:
                choice = self.alpha*choice + (1-self.alpha)*self.old_params[name]
        else:
            choice = self.old_params[name]

        self._params[name] = choice
        return choice



####################################
    #Replicating the API found in optuna: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    #copy-pasted some code
    def suggest_categorical_(self, name, choices):

        choice = self.rng.choice(choices)
        return choice

    def suggest_float_(self,
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
            value = self.rng.uniform(np.log(low),np.log(high))
            choice = np.e**value
            return choice

        else:
            if step is not None:
                choice = self.rng.choice(np.arange(low,high,step))
                return choice
            else:
                choice = self.rng.uniform(low,high)
                return choice


    def suggest_discrete_uniform_(self, name, low, high, q):
        choice = self.suggest_float(name, low, high, step=q)
        return choice


    def suggest_int_(self, name, low, high, step=1, log=False):
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
            value = self.rng.uniform(np.log(low),np.log(high))
            choice = int(np.e**value)
            return choice
        else:
            choice = self.rng.choice(list(range(low,high,step)))
            return choice

    def suggest_uniform_(self, name, low, high):
        return self.suggest_float(name, low, high)
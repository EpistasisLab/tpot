from deap import base

class ParallelToolbox(base.Toolbox):
    """Runs the TPOT genetic algorithm over multiple cores."""

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

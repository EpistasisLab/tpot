class CallBackInterface():
    def __init__(self) -> None:
        pass
    
    def step_callback(self, population):
        pass

    def population_mutate_callback(self, offspring, parent=None):
        pass

    def population_crossover_callback(self, offspring, parent=None):
        pass

    def evolutionary_algorithm_step_callback(self, population):
        pass

class Logbook():
    
    pass


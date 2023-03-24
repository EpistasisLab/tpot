import tpot2.evolutionary_algorithms.helpers as helpers
import numpy as np
import tpot2



class eaSimple_Evolver(tpot2.BaseEvolver):
    def __init__(self, 
                        selector,
                        selector_args={},
                        variation_probabilities={'mutate': 0.5,
                                                 'crossover': 0.5}, # Possible keys are: 'mutate', 'crossover', 'mutate_and_crossover', 'crossover_and_mutate'
                        **kwargs,
                        ):

        self.variation_probabilities = variation_probabilities
        self.selector=selector
        self.selector_args = selector_args
        super().__init__( **kwargs)

    def one_generation_step(self): #EA Algorithm goes here

        weighted_scores = self.population.get_column(self.population.population, column_names=self.objective_names) * self.objective_function_weights
        
        var_ops = np.random.choice(list(self.variation_probabilities.keys()), size = self.population_size, p=list(self.variation_probabilities.values()))
        
        crossover_ops = [op for op in var_ops if 'crossover' in op] # Number of operators with crossover: 'crossover', 'mutate_and_crossover', 'crossover_and_mutate'
        mutation_ops = [op for op in var_ops if op=='mutate']

        parents_index = self.selector(weighted_scores,k=len(crossover_ops), n_parents=2,   **self.selector_args)
        offspring1 = self.population.create_offspring(np.array(self.population.population)[parents_index], crossover_ops) 
        self.population.update_column(offspring1, column_names="Generation", data=self.generation, )

        parents_index = self.selector(weighted_scores,k=len(mutation_ops), n_parents=1,   **self.selector_args)
        offspring2 = self.population.create_offspring(np.array(self.population.population)[parents_index], mutation_ops) 
        self.population.update_column(offspring2, column_names="Generation", data=self.generation, )

        self.population.set_population(offspring1+offspring2)
        self.evaluate_population()



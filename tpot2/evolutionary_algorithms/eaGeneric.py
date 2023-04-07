import tpot2.evolutionary_algorithms.helpers as helpers
import numpy as np
import copy
import tpot2
import random
#based off of https://pymoo.org/algorithms/soo/ga.html



# currently each generation may evaluate fewer than the desired number of individuals if they are duplicates
class eaGeneric_Evolver(tpot2.BaseEvolver):
    def __init__(self, 
                        survival_selector,
                        parent_selector,
                        parent_selector_args = {},
                        survival_percentage = 0.5,
                        crossover_probability=.1,
                        mutation_probability=.5,
                        **kwargs,
                        ):

        
        self.survival_selector=survival_selector
        self.parent_selector=parent_selector
        self.survival_percentage = survival_percentage
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.parent_selector_args = parent_selector_args

        
        super().__init__( **kwargs)

        

    def one_generation_step(self): #EA Algorithm goes here

        self.survival_k = max(1,int(self.cur_population_size*self.survival_percentage))
        self.n_crossover = max(2,int(self.cur_population_size*self.crossover_probability))
        self.n_mutation_only = max(1,self.cur_population_size - self.n_crossover)

        #print("getting survivors")
        #Get survivors from previous 
        weighted_scores = self.population.get_column(self.population.population, column_names=self.objective_names) * self.objective_function_weights
        new_population_index = np.ravel(self.survival_selector(weighted_scores, k=self.survival_k)) #TODO make it clear that we are concatenating scores...
        self.population.set_population(np.array(self.population.population)[new_population_index])
        #print("done getting survivors")

        #print("making offspring")
        #2 parents
        weighted_scores = self.population.get_column(self.population.population, column_names=self.objective_names) * self.objective_function_weights
        #Divide n_cross_over by 2 because we generate two offspring from each pair of parents
        parents_index = self.parent_selector(weighted_scores, k=self.n_crossover, n_parents=2,   **self.parent_selector_args) #TODO make it clear that we are concatenating scores...
        var_ops = [np.random.choice(["crossover", "crossover_and_mutate"],p=[1-self.mutation_probability, self.mutation_probability]) for i in range(len(parents_index))]
        offspring = self.population.create_offspring(np.array(self.population.population)[parents_index], var_ops, n_jobs=self.n_jobs) 
        self.population.update_column(offspring, column_names="Generation", data=self.generation, )
        #print("done making offspring")

        #print("making mutations")
        #1 parent
        parents_index = self.parent_selector(weighted_scores, k=self.n_mutation_only, n_parents=1,   **self.parent_selector_args) #TODO make it clear that we are concatenating scores...
        var_ops = np.repeat("mutate",len(parents_index))
        offspring = self.population.create_offspring(np.array(self.population.population)[parents_index], var_ops) 
        self.population.update_column(offspring, column_names="Generation", data=self.generation, )
        #print("done making mutations")

        #print("evaluating")
        self.evaluate_population()
        #print("done evaluating")

        
        
       
        


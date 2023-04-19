import numpy as np
import tpot2


class SimpleEvolver(tpot2.BaseEvolver):

    def one_generation_step(self): #EA Algorithm goes here


        weighted_scores = self.population.get_column(self.population.population, column_names=self.objective_names) * self.objective_function_weights
        
        #number of crossover pairs and mutation only parent to generate
        n_crossover = int(self.cur_population_size*self.crossover_probability)
        n_crossover_then_mutate = int(self.cur_population_size*self.crossover_then_mutate_probability)
        n_mutate_then_crossover = int(self.cur_population_size*self.mutate_then_crossover_probability)
        n_total_crossover_pairs = n_crossover + n_crossover_then_mutate + n_mutate_then_crossover
        n_mutate_parents = self.cur_population_size - n_total_crossover_pairs

        #get crossover pairs
        if n_total_crossover_pairs > 0:
            cx_parents_index = self.parent_selector(weighted_scores, k=n_total_crossover_pairs, n_parents=self.n_parents,   ) #TODO make it clear that we are concatenating scores...
            cx_var_ops = np.concatenate([ np.repeat("crossover",n_crossover),
                                        np.repeat("mutate_then_crossover",n_mutate_then_crossover),
                                        np.repeat("crossover_then_mutate",n_crossover_then_mutate),
                                        ])
        else:
            cx_parents_index = []
            cx_var_ops = []
        
        #get mutation only parents
        if n_mutate_parents > 0:
            m_parents_index = self.parent_selector(weighted_scores, k=n_mutate_parents, n_parents=1,  ) #TODO make it clear that we are concatenating scores...
            m_var_ops = np.repeat("mutate",len(m_parents_index))
        else:
            m_parents_index = []
            m_var_ops = []

        cx_parents = np.array(self.population.population)[cx_parents_index]
        m_parents = np.array(self.population.population)[m_parents_index]
        parents = list(cx_parents) + list(m_parents)

        var_ops = np.concatenate([cx_var_ops, m_var_ops])
        offspring = self.population.create_offspring(parents, var_ops, n_jobs=self.n_jobs) 
        self.population.update_column(offspring, column_names="Generation", data=self.generation, )
        #print("done making offspring")

        #print("evaluating")
        self.evaluate_population()
        #print("done evaluating")

        #Get survivors from current population
        if self.survival_selector is not None:
            n_survivors = max(1,int(self.cur_population_size*self.survival_percentage)) #always keep at least one individual
            weighted_scores = self.population.get_column(self.population.population, column_names=self.objective_names) * self.objective_function_weights
            new_population_index = np.ravel(self.survival_selector(weighted_scores, k=n_survivors)) #TODO make it clear that we are concatenating scores...
            self.population.set_population(np.array(self.population.population)[new_population_index])



#All abstract methods in the Evolutionary_Optimization module

from abc import abstractmethod
import tpot2
import typing
from tqdm import tqdm, tnrange, tqdm_notebook
from tpot2.individual import BaseIndividual
import time
import numpy as np
import copy
import scipy
import os
import pickle
import statistics
from tqdm.dask import TqdmCallback
import distributed
from dask.distributed import Client
from dask.distributed import LocalCluster

class BaseEvolver():
    def __init__(   self, 
                    individual_generator ,
                    population_size: int,
                    objective_functions,
                    objective_function_weights,
                    initial_population_size = None,
                    population_scaling = .8, 
                    generations_until_end_population = 1,  
                    early_stop_tol = 0.001,
                    early_stop = None,
                    objective_names = None,
                    bigger_is_better = True,
                    verbose = 0 , 
                    callback: tpot2.CallBackInterface = None,
                    generations = 50,
                    n_jobs=1,
                    max_time_seconds=float("inf"), 
                    max_eval_time_seconds=60*5,
                    max_step_time_seconds=float("inf"), 

                    n_initial_optimizations = 0,
                    optimization_objective = None,
                    max_optimize_time_seconds=60*5,
                    optimization_steps = 10,

                    periodic_checkpoint_folder = None,

                    
                    threshold_evaluation_early_stop = None, 
                    threshold_evaluation_scaling = .2,
                    min_history_threshold = 20,
                    selection_evaluation_early_stop = None,
                    selection_evaluation_scaling = .1,
                    evalutation_early_stop_steps = None, 
                    final_score_strategy = "mean",
                    budget_range = None, 
                    budget_scaling = .8, 
                    generations_until_end_budget = 1,                    
                    stepwise_steps = 5,

                    client=None,
                    memory_limit="4GB",
                    ) -> None:
        """
        Uses mutation, crossover, and optimization functions to evolve a population of individuals towards the given objective functions.

        Parameters
        ----------
        individual_generator : generator
            Generator that yields new base individuals
        population_size : int
            Size of the population
        objective_functions : 
            list of functions that get applied to the individual and return a float or list of floats
            If an objective function returns multiple values, they are all concatenated in order 
            with respect to objective_function_weights and early_stop_tol.
        objective_function_weights : list of floats
            list of weights for each objective function. Sign flips whether bigger is better or not
        initial_population_size : int
            Size of the initial population. If None, population_size will be used.
        population_scaling : int 
            Scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
        generations_until_end_population = 1,  
            Number of generations until the population size reaches population_size
        early_stop_tol : 
            -list of floats
                list of tolerances for each objective function. If the difference between the best score and the current score is less than the tolerance, the individual is considered to have converged
                If an index of the list is None, that item will not be used for early stopping
            -int 
                If an int is given, it will be used as the tolerance for all objectives
        early_stop : int
            Number of generations without improvement before early stopping. All objectives must have converged within the tolerance for this to be triggered.
        objective_names : list of strings
            Names of the objectives. If None, objective0, objective1, etc. will be used
        bigger_is_better : bool
            If True, bigger is better for all objectives. If False, smaller is better for all objectives. This applies to the weighted scores.
        - verbose (int): How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar
            2. evaluations progress bar
            3. best individual
            4. warnings
            5. full warnings trace
        callback : tpot2.CallBackInterface
            Callback object. Not implemented
        generations : int
            Number of generations to run
        n_jobs : int
            Number of jobs to run in parallel
        max_time_seconds : float
            Maximum time to run the optimization
        max_eval_time_seconds : float
            Maximum time to evaluate a single individual
        max_step_time_seconds : float
            Maximum time to run a single step. Not implemented
        n_initial_optimizations : int
            Number of individuals to optimize before starting the evolution.
        optimization_objective : function
            Function to optimize the individual with. If None, the first objective function will be used
        max_optimize_time_seconds : float
            Maximum time to run an optimization
        optimization_steps : int
            Number of steps per optimization
        periodic_checkpoint_folder : str
            Folder to save the population to periodically. If None, no periodic saving will be done. Will save once every generation but not more than once every 10 minutes.
            If provided, training will resume from this checkpoint.
        - threshold_evaluation_early_stop (list): [start, end] A starting and ending percentile to use as a threshold for the evaluation early stopping.
            Values between 0 and 100.
        - threshold_evaluation_scaling (float): [0,inf) A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        
        - min_history_threshold (int): The minimum number of previous scores needed before using threshold early stopping.
        
        - selection_evaluation_early_stop (list): A lower and upper percent of the population size to select each round of CV.
            Values between 0 and 1.
        
        - selection_evaluation_scaling (float): A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        -evalutation_early_stop_steps (int): The number of steps that will be taken from the objective function. (e.g., the number of CV folds to evaluate)

        - final_score_strategy (str): The strategy to use when determining the final score for an individual.
            "mean": The mean of all objective scores
            "last": The score returned by the last call. Currently each objective is evalauted with a clone of the individual.

        - budget_range (list): [start, end] A starting and ending budget to use for the budget scaling.
        
        - budget_scaling (float): [0,1] A scaling factor to use when determining how fast we move the budget from the start to end budget.

        - generations_until_max_budget (int): The number of generations to run before reaching the max budget.

        - stepwise_steps (int): The number of staircase steps to take when scaling the budget and population size.

        - client (dask.distributed.Client): A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client. 

        - memory_limit (str): The maximum amount of memory that the optimization process should use per thread. See https://docs.dask.org/en/stable/deploying-python.html
        """


        if threshold_evaluation_early_stop is not None or selection_evaluation_early_stop is not None:
            if evalutation_early_stop_steps is None:
                raise ValueError("evalutation_early_stop_steps must be set when using threshold_evaluation_early_stop or selection_evaluation_early_stop")

        self.individual_generator = individual_generator 
        self.population_size = population_size 
        self.objective_functions = objective_functions 
        self.objective_function_weights = np.array(objective_function_weights)
        self.bigger_is_better = bigger_is_better
        if not bigger_is_better:
            self.objective_function_weights = np.array(self.objective_function_weights)*-1

        self.initial_population_size = initial_population_size
        if self.initial_population_size is None:
            self.cur_population_size = population_size
        else:
            self.cur_population_size = initial_population_size

        self.population_scaling = population_scaling
        self.generations_until_end_population = generations_until_end_population

        self.population_size_list = None


        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.verbose  = verbose  
        self.callback = callback 
        self.generations = generations 
        self.n_jobs = n_jobs
        self.max_time_seconds = max_time_seconds  
        self.max_eval_time_seconds = max_eval_time_seconds 

        self.max_step_time_seconds = max_step_time_seconds
        
        self.n_initial_optimizations  = n_initial_optimizations  
        self.optimization_objective  = optimization_objective  
        self.max_optimize_time_seconds = max_optimize_time_seconds 
        self.optimization_steps = optimization_steps 
        
        self.generation = 0


        self.threshold_evaluation_early_stop =threshold_evaluation_early_stop
        self.threshold_evaluation_scaling =  max(0.00001,threshold_evaluation_scaling ) 
        self.min_history_threshold = min_history_threshold

        self.selection_evaluation_early_stop = selection_evaluation_early_stop
        self.selection_evaluation_scaling =  max(0.00001,selection_evaluation_scaling )
        self.evalutation_early_stop_steps = evalutation_early_stop_steps
        self.final_score_strategy = final_score_strategy

        self.budget_range = budget_range
        self.budget_scaling = budget_scaling
        self.generations_until_end_budget = generations_until_end_budget
        self.stepwise_steps = stepwise_steps

        self.memory_limit = memory_limit

        self.client = client



        ###########

        if self.initial_population_size != self.population_size:
            self.population_size_list = beta_interpolation(start=self.cur_population_size, end=self.population_size, scale=self.population_scaling, n=generations_until_end_population, n_steps=self.stepwise_steps)
            self.population_size_list = np.round(self.population_size_list).astype(int)

        if self.budget_range is None:
            self.budget_list = None
        else:
            self.budget_list = beta_interpolation(start=self.budget_range[0], end=self.budget_range[1], n=self.generations_until_end_budget, scale=self.budget_scaling, n_steps=self.stepwise_steps)

        if objective_names is None:
            self.objective_names = ["objective"+str(i) for i in range(len(objective_function_weights))]
        else:
            self.objective_names = objective_names

        if self.budget_list is not None:
            if len(self.budget_list) <= self.generation:
                self.budget = self.budget_list[-1]
            else:
                self.budget = self.budget_list[self.generation]
        else:
            self.budget = None
        

        self.early_stop_tol = early_stop_tol
        self.early_stop = early_stop

        if isinstance(self.early_stop_tol, float):
            self.early_stop_tol = [self.early_stop_tol for _ in range(len(self.objective_names))]

        self.early_stop_tol = [np.inf if tol is None else tol for tol in self.early_stop_tol]

        self.population = None
        self.population_file = None
        if self.periodic_checkpoint_folder is not None:
            self.population_file = os.path.join(self.periodic_checkpoint_folder, "population.pkl")
            if not os.path.exists(self.periodic_checkpoint_folder):
                os.makedirs(self.periodic_checkpoint_folder)
            if os.path.exists(self.population_file):
                self.population = pickle.load(open(self.population_file, "rb"))

                if len(self.population.evaluated_individuals)>0 and "Generation" in self.population.evaluated_individuals.columns:      
                    self.generation = self.population.evaluated_individuals['Generation'].max() + 1 #TODO check if this is empty?

        init_names = self.objective_names
        if self.budget_range is not None:
            init_names = init_names + ["budget"]
        if self.population is None:
            self.population = tpot2.Population(column_names=init_names)
            initial_population = [next(self.individual_generator) for _ in range(self.cur_population_size)]
            self.population.add_to_population(initial_population)
            self.population.update_column(self.population.population, column_names="Generation", data=self.generation)


    def optimize(self, generations=None):

        if self.client is not None: #If user passed in a client manually
           self._client = self.client
        else:

            if self.verbose >= 4:
                silence_logs = 30
            elif self.verbose >=5:
                silence_logs = 40
            else:
                silence_logs = 50
            self._cluster = LocalCluster(n_workers=self.n_jobs, #if no client is passed in and no global client exists, create our own
                    threads_per_worker=1,
                    silence_logs=silence_logs,
                    processes=False,
                    memory_limit=self.memory_limit)
            self._client = Client(self._cluster)
        

        if self.n_initial_optimizations > 0:
            self.optimize_population()
        if generations is None:
            generations = self.generations

        start_time = time.time() 
        
        last_save_time = time.time()
        
        generations_without_improvement = np.array([0 for _ in range(len(self.objective_function_weights))])
        best_scores = [-np.inf for _ in range(len(self.objective_function_weights))]

        try: 
            for gen in tnrange(generations,desc="Generation", disable=self.verbose<1):
                
                # Generation 0 is the initial population
                if self.generation == 0:
                    if self.population_file is not None:
                        pickle.dump(self.population, open(self.population_file, "wb"))
                    self.evaluate_population()
                    if self.population_file is not None:
                        pickle.dump(self.population, open(self.population_file, "wb"))
                    
                    attempts = 2
                    while len(self.population.population) == 0 and attempts > 0:
                        new_initial_population = [next(self.individual_generator) for _ in range(self.cur_population_size)]
                        self.population.add_to_population(new_initial_population)
                        attempts -= 1
                        self.evaluate_population()

                    if len(self.population.population) == 0:
                        raise Exception("No individuals could be evaluated in the initial population")

                    self.generation += 1
                # Generation 1 is the first generation after the initial population
                else:
                    if time.time() - start_time > self.max_time_seconds:
                        break
                    self.step()

                if self.verbose >= 3:  
                    sign = np.sign(self.objective_function_weights)
                    valid_df = self.population.evaluated_individuals[~self.population.evaluated_individuals[self.objective_names].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names]*sign
                    cur_best_scores = valid_df.max(axis=0)*sign
                    cur_best_scores = cur_best_scores.to_numpy()
                    print("Generation: ", self.generation)
                    for i, obj in enumerate(self.objective_names):
                        print(f"Best {obj} score: {cur_best_scores[i]}")


                if self.early_stop:
                    if self.budget is None or self.budget>=1:
                        #get sign of objective_function_weights
                        sign = np.sign(self.objective_function_weights)
                        #get best score for each objective
                        valid_df = self.population.evaluated_individuals[~self.population.evaluated_individuals[self.objective_names].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names]*sign
                        cur_best_scores = valid_df.max(axis=0)*sign
                        cur_best_scores = cur_best_scores.to_numpy()
                        #cur_best_scores =  self.population.get_column(self.population.population, column_names=self.objective_names).max(axis=0)*sign #TODO this assumes the current population is the best
                        
                        improved = ( np.array(best_scores) - np.array(cur_best_scores) <= np.array(self.early_stop_tol) )
                        not_improved = np.logical_not(improved)
                        generations_without_improvement = generations_without_improvement* not_improved + not_improved #set to zero if not improved, else increment

                        #update best score
                        best_scores = [max(best_scores[i], cur_best_scores[i]) for i in range(len(self.objective_names))]

                        if all(generations_without_improvement>self.early_stop):
                            if self.verbose >= 3:
                                print("Early stop")
                            break

                #save population
                if self.population_file is not None: # and time.time() - last_save_time > 60*10:
                    pickle.dump(self.population, open(self.population_file, "wb"))

        except KeyboardInterrupt:
            if self.verbose >= 3:
                print("KeyboardInterrupt")
            
            self.population.remove_invalid_from_population(column_names=self.objective_names, invalid_value="INVALID")
            self.population.remove_invalid_from_population(column_names=self.objective_names, invalid_value="TIMEOUT")


        

        if self.population_file is not None:
            pickle.dump(self.population, open(self.population_file, "wb"))

        if self.client is None: #If we created our own client, close it
            self._client.close()
            self._cluster.close()


    def step(self,):
        if self.population_size_list is not None:
            if self.generation < len(self.population_size_list):
                self.cur_population_size = self.population_size_list[self.generation]
            else:
                self.cur_population_size = self.population_size

                
        if self.budget_list is not None:
            if len(self.budget_list) <= self.generation:
                self.budget = self.budget_range[-1]
            else:
                self.budget = self.budget_list[self.generation]
        else:
            self.budget = None



        self.one_generation_step()
        self.generation += 1
        

    @abstractmethod
    def one_generation_step(self, ): #your EA Algorithm goes here
        
        #generate new individuals

        #evaluate new individuals

        #add to population

        #remove individuals
        
        pass
    
    def optimize_population(self,):
        individuals_to_optimize = [copy.deepcopy(ind) for ind in self.population.population[0:self.n_initial_optimizations]]
        tpot2.objectives.parallel_optimize_objective(individuals_to_optimize, self.optimization_objective, self.n_jobs, verbose=self.verbose, timeout=self.max_optimize_time_seconds)
        self.population.set_population(individuals_to_optimize)
        #self.population.update_log_list(individuals_to_optimize, scores, column_name="scores")
        #self.population.remove_invalid_from_population(column_name="scores")

    # Gets a list of unevaluated individuals in the livepopulation, evaluates them, and removes failed attempts
    # TODO This could probably be an independent function?
    def evaluate_population(self,):
        
        #Update the sliding scales and thresholds 
        # Save population, TODO remove some of these
        if self.population_file is not None: # and time.time() - last_save_time > 60*10:
            pickle.dump(self.population, open(self.population_file, "wb"))
            last_save_time = time.time()


        #Get the current thresholds per step
        self.thresholds = None
        if self.threshold_evaluation_early_stop is not None:
            old_data = self.population.evaluated_individuals[self.objective_names]
            old_data = old_data[old_data[self.objective_names].notnull().all(axis=1)]
            if len(old_data) >= self.min_history_threshold:
                self.thresholds = np.array([get_thresholds(old_data[obj_name],
                                                            start=self.threshold_evaluation_early_stop[0], 
                                                            end=self.threshold_evaluation_early_stop[1], 
                                                            scale=self.threshold_evaluation_scaling,
                                                            n=self.evalutation_early_stop_steps)
                                        for obj_name in self.objective_names]).T

        #Get the selectors survival rates per step
        if self.selection_evaluation_early_stop is not None:
            lower = self.selection_evaluation_early_stop[0]
            upper = self.selection_evaluation_early_stop[1]
            survival_counts = self.cur_population_size*(scipy.special.betainc(1,self.threshold_evaluation_scaling,np.linspace(0,1,self.evalutation_early_stop_steps))*(upper-lower)+lower)
            self.survival_counts = survival_counts.astype(int)
        else:
            self.survival_counts = None



        if self.evalutation_early_stop_steps is not None:
            if self.survival_counts is None:
                #TODO if we are not using selection method for each step, we can create single threads that run all steps for an individual. No need to come back each step.
                self.evaluate_population_selection_early_stop(survival_counts=self.survival_counts, thresholds=self.thresholds, budget=self.budget)
            else:
                #parallelize one step at a time. After each step, come together and select the next individuals to run the next step on.
                self.evaluate_population_selection_early_stop(survival_counts=self.survival_counts, thresholds=self.thresholds, budget=self.budget)
        else:
            self.evaluate_population_full(budget=self.budget)


        # Save population, TODO remove some of these
        if self.population_file is not None: # and time.time() - last_save_time > 60*10:
            pickle.dump(self.population, open(self.population_file, "wb"))
            last_save_time = time.time()

    def evaluate_population_full(self, budget=None):
        individuals_to_evaluate = self.get_unevaluated_individuals(self.objective_names, budget=budget,)
        
        #print("evaluating this many individuals: ", len(individuals_to_evaluate))

        if len(individuals_to_evaluate) == 0:
            if self.verbose > 3:
                print("No new individuals to evaluate")
            return

        scores = tpot2.objectives.parallel_eval_objective_list(individuals_to_evaluate, self.objective_functions, self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds, budget=budget, n_expected_columns=len(self.objective_names), client=self._client )


        self.population.update_column(individuals_to_evaluate, column_names=self.objective_names, data=scores)
        if budget is not None:
            self.population.update_column(individuals_to_evaluate, column_names="budget", data=budget)

        self.population.remove_invalid_from_population(column_names=self.objective_names)
        self.population.remove_invalid_from_population(column_names=self.objective_names, invalid_value="TIMEOUT")

    def get_unevaluated_individuals(self, column_names, budget=None, individual_list=None):
        if individual_list is not None:
            cur_pop = np.array(individual_list)
        else:
            cur_pop = np.array(self.population.population)

        if all([name_step in self.population.evaluated_individuals.columns for name_step in column_names]):
            if budget is not None:
                offspring_scores = self.population.get_column(cur_pop, column_names=column_names+["budget"], to_numpy=False)
                #Individuals are unevaluated if we have a higher budget OR if any of the objectives are nan
                unevaluated_filter = lambda i: any(offspring_scores.loc[offspring_scores.index[i]][column_names].isna()) or (offspring_scores.loc[offspring_scores.index[i]]["budget"] < budget)
            else:
                offspring_scores = self.population.get_column(cur_pop, column_names=column_names, to_numpy=False)
                unevaluated_filter = lambda i: any(offspring_scores.loc[offspring_scores.index[i]][column_names].isna())
            unevaluated_individuals_this_step = [i for i in range(len(cur_pop)) if unevaluated_filter(i)]
            return cur_pop[unevaluated_individuals_this_step]
        
        else: #if column names are not in the evaluated_individuals, then we have not evaluated any individuals yet
            for name_step in column_names:
                self.population.evaluated_individuals[name_step] = np.nan
            return cur_pop

    def evaluate_population_selection_early_stop(self,survival_counts, thresholds=None, budget=None):


        survival_selector = tpot2.evolutionary_algorithms.parent_selectors.survival_select_NSGA2

        ################

        objective_function_signs = np.sign(self.objective_function_weights)


        cur_individuals = self.population.population.copy()
        
        all_step_names = []
        for step in range(self.evalutation_early_stop_steps):
            if budget is None:
                this_step_names = [f"{n}_step_{step}" for n in self.objective_names]
            else:
                this_step_names = [f"{n}_budget_{budget}_step_{step}" for n in self.objective_names]
            
            all_step_names.append(this_step_names)
            
            unevaluated_individuals_this_step = self.get_unevaluated_individuals(this_step_names, budget=None, individual_list=cur_individuals)

            if len(unevaluated_individuals_this_step) == 0:
                if self.verbose > 3:
                    print("No new individuals to evaluate")
                continue
            
            scores = tpot2.objectives.parallel_eval_objective_list(individual_list=unevaluated_individuals_this_step,
                                    objective_list=self.objective_functions,
                                    n_jobs = self.n_jobs,
                                    verbose=self.verbose,
                                    timeout=self.max_eval_time_seconds,
                                    step=step,
                                    budget = self.budget,
                                    generation = self.generation,
                                    n_expected_columns=len(self.objective_names),
                                    client=self._client,
                                    )

            self.population.update_column(unevaluated_individuals_this_step, column_names=this_step_names, data=scores)

            self.population.remove_invalid_from_population(column_names=this_step_names)
            self.population.remove_invalid_from_population(column_names=this_step_names, invalid_value="TIMEOUT")

            #remove invalids:
            invalids = []
            #find indeces of invalids

            for j in range(len(scores)):
                if  any([s=="INVALID" for s in scores[j]]):
                    invalids.append(j)

            for j in range(len(scores)):
                if  any([s=="TIMEOUT" for s in scores[j]]):
                    invalids.append(j)


            #already evaluated
            already_evaluated = list(set(cur_individuals) - set(unevaluated_individuals_this_step))
            #evaluated and valid
            valid_evaluations_this_step = remove_items(unevaluated_individuals_this_step,invalids)
            #update cur_individuals with current individuals with valid scores
            cur_individuals = np.concatenate([already_evaluated, valid_evaluations_this_step])


            #Get average scores

            #array of shape (steps, individuals, objectives)
            offspring_scores = [self.population.get_column(cur_individuals, column_names=step_names) for step_names in all_step_names]
            offspring_scores = np.array(offspring_scores)
            if self.final_score_strategy == 'mean':
                offspring_scores  = offspring_scores.mean(axis=0)
            elif self.final_score_strategy == 'last':
                offspring_scores = offspring_scores[-1]


            #if last step, add the final metrics
            if step == self.evalutation_early_stop_steps-1:
                self.population.update_column(cur_individuals, column_names=self.objective_names, data=offspring_scores)
                if budget is not None:
                    self.population.update_column(cur_individuals, column_names="budget", data=budget)
                return

            #If we have more threads than remaining individuals, we may as well evaluate the extras too
            if self.n_jobs < len(cur_individuals):
                #Remove based on thresholds
                if thresholds is not None:
                    threshold = thresholds[step]
                    invalids = []
                    for i in range(len(offspring_scores)):

                        if all([s*w>t*w for s,t,w in zip(offspring_scores[i],threshold,objective_function_signs)  ]):
                            invalids.append(i)

                    if len(invalids) > 0:
                        
                        max_to_remove = min(len(cur_individuals) - self.n_jobs, len(invalids))
                        
                        if max_to_remove < len(invalids):
                            invalids = np.random.choice(invalids, max_to_remove, replace=False)

                        cur_individuals = remove_items(cur_individuals,invalids)
                        offspring_scores = remove_items(offspring_scores,invalids)

                #Remove based on selection
                if step < self.evalutation_early_stop_steps - 1 and survival_counts[step]>1: #don't do selection for the last loop since they are completed
                    k = survival_counts[step] + len(invalids) #TODO can remove the min if the selections method can ignore k>population size
                    if len(cur_individuals)> 1 and k > self.n_jobs and k < len(cur_individuals):
                        weighted_scores = np.array([s * self.objective_function_weights for s in offspring_scores ])

                        new_population_index = survival_selector(weighted_scores, k=k)
                        cur_individuals = np.array(cur_individuals)[new_population_index]

                    

        









def get_thresholds(scores, start=0, end=1, scale=.5, n=10,):
       thresh = beta_interpolation(start=start, end=end, scale=scale, n=n)
       return [np.percentile(scores, t) for t in thresh]





def equalize_list(lst, n_steps):
    step_size = len(lst) / n_steps
    new_lst = []
    for i in range(n_steps):
        start_index = int(i * step_size)
        end_index = int((i+1) * step_size)
        if i == 0: # First segment
            step_lst = [lst[start_index]] * (end_index - start_index)
        elif i == n_steps-1: # Last segment
            step_lst = [lst[-1]] * (end_index - start_index)
        else: # Middle segment
            segment = lst[start_index:end_index]
            median_value = statistics.median(segment)
            step_lst = [median_value] * (end_index - start_index)
        new_lst.extend(step_lst)
    return new_lst


def beta_interpolation(start=0, end=1, scale=1, n=10, n_steps=None):
    if n_steps is None:
        n_steps = n
    if n_steps > n:
        n_steps = n
    if scale <= 0:
        scale = 0.0001
    if scale >= 1:
        scale = 0.9999

    alpha = 2 * scale
    beta = 2 - alpha
    x = np.linspace(0,1,n)
    values = scipy.special.betainc(alpha,beta,x)*(end-start)+start

    if n_steps is not None:
        return equalize_list(values, n_steps)
    else:
        return values

#thanks chat gtp
def remove_items(items, indexes_to_remove):
    items = items.copy()
    #if items is a numpy array, we need to convert to a list
    if type(items) == np.ndarray:
        items = items.tolist()
    for index in sorted(indexes_to_remove, reverse=True):
        del items[index]
    return np.array(items)
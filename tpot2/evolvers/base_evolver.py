#All abstract methods in the Evolutionary_Optimization module

from abc import abstractmethod
import tpot2
import typing
import tqdm
from tpot2 import BaseIndividual
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
from tpot2.selectors import survival_select_NSGA2, tournament_selection_dominated
import math
from tpot2.utils.utils import get_thresholds, beta_interpolation, remove_items, equalize_list

def ind_mutate(ind, rng):
    rng = np.random.default_rng(rng)
    return ind.mutate(rng=rng)

def ind_crossover(ind1, ind2, rng):
    rng = np.random.default_rng(rng)
    return ind1.crossover(ind2, rng=rng)

class BaseEvolver():
    def __init__(   self,
                    individual_generator ,

                    objective_functions,
                    objective_function_weights,
                    objective_names = None,
                    objective_kwargs = None,
                    bigger_is_better = True,

                    population_size = 50,
                    initial_population_size = None,
                    population_scaling = .5,
                    generations_until_end_population = 1,
                    generations = 50,
                    early_stop = None,
                    early_stop_tol = 0.001,


                    max_time_seconds=float("inf"),
                    max_eval_time_seconds=60*5,

                    n_jobs=1,
                    memory_limit="4GB",
                    client=None,

                    survival_percentage = 1,
                    crossover_probability=.2,
                    mutate_probability=.7,
                    mutate_then_crossover_probability=.05,
                    crossover_then_mutate_probability=.05,

                    mutation_functions = [ind_mutate],
                    crossover_functions = [ind_crossover],

                    mutation_function_weights = None,
                    crossover_function_weights = None,

                    n_parents=2,

                    survival_selector = survival_select_NSGA2,
                    parent_selector = tournament_selection_dominated,

                    budget_range = None,
                    budget_scaling = .5,
                    generations_until_end_budget = 1,
                    stepwise_steps = 5,

                    threshold_evaluation_early_stop = None,
                    threshold_evaluation_scaling = .5,
                    min_history_threshold = 20,
                    selection_evaluation_early_stop = None,
                    selection_evaluation_scaling = .5,
                    evaluation_early_stop_steps = None,
                    final_score_strategy = "mean",

                    verbose = 0,
                    periodic_checkpoint_folder = None,
                    callback = None,
                    rng=None,

                    ) -> None:
        """
        Uses mutation, crossover, and optimization functions to evolve a population of individuals towards the given objective functions.

        Parameters
        ----------
        individual_generator : generator
            Generator that yields new base individuals. Used to generate initial population.
        objective_functions : list of callables
            list of functions that get applied to the individual and return a float or list of floats
            If an objective function returns multiple values, they are all concatenated in order
            with respect to objective_function_weights and early_stop_tol.
        objective_function_weights : list of floats
            list of weights for each objective function. Sign flips whether bigger is better or not
        objective_names : list of strings, default=None
            Names of the objectives. If None, objective0, objective1, etc. will be used
        objective_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to the objective function
        bigger_is_better : bool, default=True
            If True, the objective function is maximized. If False, the objective function is minimized. Use negative weights to reverse the direction.
        population_size : int, default=50
            Size of the population
        initial_population_size : int, default=None
            Size of the initial population. If None, population_size will be used.
        population_scaling : int, default=0.5
            Scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
        generations_until_end_population : int, default=1
            Number of generations until the population size reaches population_size
        generations : int, default=50
            Number of generations to run
        early_stop : int, default=None
            Number of generations without improvement before early stopping. All objectives must have converged within the tolerance for this to be triggered.
        early_stop_tol : float, list of floats, or None, default=0.001
            -list of floats
                list of tolerances for each objective function. If the difference between the best score and the current score is less than the tolerance, the individual is considered to have converged
                If an index of the list is None, that item will not be used for early stopping
            -int
                If an int is given, it will be used as the tolerance for all objectives
        max_time_seconds : float, default=float("inf")
            Maximum time to run the optimization. If none or inf, will run until the end of the generations.
        max_eval_time_seconds : float, default=60*5
            Maximum time to evaluate a single individual. If none or inf, there will be no time limit per evaluation.
        n_jobs : int, default=1
            Number of processes to run in parallel.
        memory_limit : str, default="4GB"
            Memory limit for each job. See Dask [LocalCluster documentation](https://distributed.dask.org/en/stable/api.html#distributed.Client) for more information.
        client : dask.distributed.Client, default=None
            A dask client to use for parallelization. If not None, this will override the n_jobs and memory_limit parameters. If None, will create a new client with num_workers=n_jobs and memory_limit=memory_limit.
        survival_percentage : float, default=1
            Percentage of the population size to utilize for mutation and crossover at the beginning of the generation. The rest are discarded. Individuals are selected with the selector passed into survival_selector. The value of this parameter must be between 0 and 1, inclusive.
            For example, if the population size is 100 and the survival percentage is .5, 50 individuals will be selected with NSGA2 from the existing population. These will be used for mutation and crossover to generate the next 100 individuals for the next generation. The remainder are discarded from the live population. In the next generation, there will now be the 50 parents + the 100 individuals for a total of 150. Surivival percentage is based of the population size parameter and not the existing population size (current population size when using successive halving). Therefore, in the next generation we will still select 50 individuals from the currently existing 150.
        crossover_probability : float, default=.2
            Probability of generating a new individual by crossover between two individuals.
        mutate_probability : float, default=.7
            Probability of generating a new individual by crossover between one individuals.
        mutate_then_crossover_probability : float, default=.05
            Probability of generating a new individual by mutating two individuals followed by crossover.
        crossover_then_mutate_probability : float, default=.05
            Probability of generating a new individual by crossover between two individuals followed by a mutation of the resulting individual.
        n_parents : int, default=2
            Number of parents to use for crossover. Must be greater than 1.
        survival_selector : function, default=survival_select_NSGA2
            Function to use to select individuals for survival. Must take a matrix of scores and return selected indexes.
            Used to selected population_size * survival_percentage individuals at the start of each generation to use for mutation and crossover.
        parent_selector : function, default=parent_select_NSGA2
            Function to use to select pairs parents for crossover and individuals for mutation. Must take a matrix of scores and return selected indexes.
        budget_range : list [start, end], default=None
            A starting and ending budget to use for the budget scaling.
        budget_scaling float : [0,1], default=0.5
            A scaling factor to use when determining how fast we move the budget from the start to end budget.
        generations_until_end_budget : int, default=1
            The number of generations to run before reaching the max budget.
        stepwise_steps : int, default=1
            The number of staircase steps to take when scaling the budget and population size.
        threshold_evaluation_early_stop : list [start, end], default=None
            starting and ending percentile to use as a threshold for the evaluation early stopping.
            Values between 0 and 100.
        threshold_evaluation_scaling : float [0,inf), default=0.5
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        min_history_threshold : int, default=0
            The minimum number of previous scores needed before using threshold early stopping.
        selection_evaluation_early_stop : list, default=None
            A lower and upper percent of the population size to select each round of CV.
            Values between 0 and 1.
        selection_evaluation_scaling : float, default=0.5
            A scaling factor to use when determining how fast we move the threshold moves from the start to end percentile.
            Must be greater than zero. Higher numbers will move the threshold to the end faster.
        evaluation_early_stop_steps : int, default=1
            The number of steps that will be taken from the objective function. (e.g., the number of CV folds to evaluate)
        final_score_strategy : str, default="mean"
            The strategy to use when determining the final score for an individual.
            "mean": The mean of all objective scores
            "last": The score returned by the last call. Currently each objective is evaluated with a clone of the individual.
        verbose : int, default=0
            How much information to print during the optimization process. Higher values include the information from lower values.
            0. nothing
            1. progress bar
            2. evaluations progress bar
            3. best individual
            4. warnings
            >=5. full warnings trace
        periodic_checkpoint_folder : str, default=None
            Folder to save the population to periodically. If None, no periodic saving will be done.
            If provided, training will resume from this checkpoint.
        callback : tpot2.CallBackInterface, default=None
            Callback object. Not implemented
        rng : Numpy.Random.Generator, None, default=None
            An object for reproducability of experiments. This value will be passed to numpy.random.default_rng() to create an instnce of the genrator to pass to other classes

            - Numpy.Random.Generator
                Will be used to create and lock in Generator instance with 'numpy.random.default_rng()'. Note this will be the same Generator passed in.
            - None
                Will be used to create Generator for 'numpy.random.default_rng()' where a fresh, unpredictable entropy will be pulled from the OS
        """

        self.rng = np.random.default_rng(rng)

        if threshold_evaluation_early_stop is not None or selection_evaluation_early_stop is not None:
            if evaluation_early_stop_steps is None:
                raise ValueError("evaluation_early_stop_steps must be set when using threshold_evaluation_early_stop or selection_evaluation_early_stop")

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



        if max_time_seconds is None:
            self.max_time_seconds = float("inf")
        else:
            self.max_time_seconds = max_time_seconds

        #functools requires none for infinite time, doesn't support inf
        if max_eval_time_seconds is not None and math.isinf(max_eval_time_seconds ):
            self.max_eval_time_seconds = None
        else:
            self.max_eval_time_seconds = max_eval_time_seconds




        self.generation = 0


        self.threshold_evaluation_early_stop =threshold_evaluation_early_stop
        self.threshold_evaluation_scaling =  max(0.00001,threshold_evaluation_scaling )
        self.min_history_threshold = min_history_threshold

        self.selection_evaluation_early_stop = selection_evaluation_early_stop
        self.selection_evaluation_scaling =  max(0.00001,selection_evaluation_scaling )
        self.evaluation_early_stop_steps = evaluation_early_stop_steps
        self.final_score_strategy = final_score_strategy

        self.budget_range = budget_range
        self.budget_scaling = budget_scaling
        self.generations_until_end_budget = generations_until_end_budget
        self.stepwise_steps = stepwise_steps

        self.memory_limit = memory_limit

        self.client = client


        self.survival_selector=survival_selector
        self.parent_selector=parent_selector
        self.survival_percentage = survival_percentage

        total_var_p = crossover_probability + mutate_probability + mutate_then_crossover_probability + crossover_then_mutate_probability
        self.crossover_probability = crossover_probability / total_var_p
        self.mutate_probability = mutate_probability  / total_var_p
        self.mutate_then_crossover_probability= mutate_then_crossover_probability / total_var_p
        self.crossover_then_mutate_probability= crossover_then_mutate_probability / total_var_p


        self.mutation_functions = mutation_functions
        self.crossover_functions = crossover_functions

        if mutation_function_weights is None:
            self.mutation_function_weights = [1 for _ in range(len(mutation_functions))]
        else:
            self.mutation_function_weights = mutation_function_weights

        if mutation_function_weights is None:
            self.crossover_function_weights = [1 for _ in range(len(mutation_functions))]
        else:
            self.crossover_function_weights = crossover_function_weights

        self.n_parents = n_parents

        if objective_kwargs is None:
            self.objective_kwargs = {}
        else:
            self.objective_kwargs = objective_kwargs

        # if objective_kwargs is None:
        #     self.objective_kwargs = [{}] * len(self.objective_functions)
        # elif isinstance(objective_kwargs, dict):
        #     self.objective_kwargs = [objective_kwargs] * len(self.objective_functions)
        # else:
        #     self.objective_kwargs = objective_kwargs

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
            init_names = init_names + ["Budget"]
        if self.population is None:
            self.population = tpot2.Population(column_names=init_names)
            initial_population = [next(self.individual_generator) for _ in range(self.cur_population_size)]
            self.population.add_to_population(initial_population, self.rng)
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
                    processes=True,
                    memory_limit=self.memory_limit)
            self._client = Client(self._cluster)



        if generations is None:
            generations = self.generations

        start_time = time.time()

        generations_without_improvement = np.array([0 for _ in range(len(self.objective_function_weights))])
        best_scores = [-np.inf for _ in range(len(self.objective_function_weights))]


        self.scheduled_timeout_time = time.time() + self.max_time_seconds


        try:
            #for gen in tnrange(generations,desc="Generation", disable=self.verbose<1):
            done = False
            gen = 0
            if self.verbose >= 1:
                if generations is None or np.isinf(generations):
                    pbar = tqdm.tqdm(total=0)
                else:
                    pbar = tqdm.tqdm(total=generations)
                pbar.set_description("Generation")
            while not done:
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
                        raise Exception("No individuals could be evaluated in the initial population. This may indicate a bug in the configuration, included models, or objective functions. Set verbose>=4 to see the errors that caused individuals to fail.")

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
                    if self.budget is None or self.budget>=self.budget_range[-1]: #self.budget>=1:
                        #get sign of objective_function_weights
                        sign = np.sign(self.objective_function_weights)
                        #get best score for each objective
                        valid_df = self.population.evaluated_individuals[~self.population.evaluated_individuals[self.objective_names].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names]*sign
                        cur_best_scores = valid_df.max(axis=0)
                        cur_best_scores = cur_best_scores.to_numpy()
                        #cur_best_scores =  self.population.get_column(self.population.population, column_names=self.objective_names).max(axis=0)*sign #TODO this assumes the current population is the best

                        improved = ( np.array(cur_best_scores) - np.array(best_scores) >= np.array(self.early_stop_tol) )
                        not_improved = np.logical_not(improved)
                        generations_without_improvement = generations_without_improvement * not_improved + not_improved #set to zero if not improved, else increment
                        pass
                        #update best score
                        best_scores = [max(best_scores[i], cur_best_scores[i]) for i in range(len(self.objective_names))]

                        if all(generations_without_improvement>self.early_stop):
                            if self.verbose >= 3:
                                print("Early stop")
                            break

                #save population
                if self.population_file is not None: # and time.time() - last_save_time > 60*10:
                    pickle.dump(self.population, open(self.population_file, "wb"))

                gen += 1
                if self.verbose >= 1:
                    pbar.update(1)

                if generations is not None and gen >= generations:
                    done = True

        except KeyboardInterrupt:
            if self.verbose >= 3:
                print("KeyboardInterrupt")
            self.population.remove_invalid_from_population(column_names=self.objective_names, invalid_value="INVALID")
            self.population.remove_invalid_from_population(column_names=self.objective_names, invalid_value="TIMEOUT")
            self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="INVALID")
            self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="TIMEOUT")




        if self.population_file is not None:
            pickle.dump(self.population, open(self.population_file, "wb"))

        if self.client is None: #If we created our own client, close it
            self._client.close()
            self._cluster.close()

        tpot2.utils.get_pareto_frontier(self.population.evaluated_individuals, column_names=self.objective_names, weights=self.objective_function_weights)

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

        if self.survival_selector is not None:
            n_survivors = max(1,int(self.cur_population_size*self.survival_percentage)) #always keep at least one individual
            self.population.survival_select(    selector=self.survival_selector,
                                                weights=self.objective_function_weights,
                                                columns_names=self.objective_names,
                                                n_survivors=n_survivors,
                                                inplace=True,
                                                rng=self.rng,)

        self.generate_offspring()
        self.evaluate_population()

        self.generation += 1

    def generate_offspring(self, ): #your EA Algorithm goes here
        parents = self.population.parent_select(selector=self.parent_selector, weights=self.objective_function_weights, columns_names=self.objective_names, k=self.cur_population_size, n_parents=2, rng=self.rng)
        p = np.array([self.crossover_probability, self.mutate_then_crossover_probability, self.crossover_then_mutate_probability, self.mutate_probability])
        p = p / p.sum()
        var_op_list = self.rng.choice(["crossover", "mutate_then_crossover", "crossover_then_mutate", "mutate"], size=self.cur_population_size, p=p)

        for i, op in enumerate(var_op_list):
            if op == "mutate":
                parents[i] = parents[i][0] #mutations take a single individual

        offspring = self.population.create_offspring2(parents, var_op_list, self.mutation_functions, self.mutation_function_weights, self.crossover_functions, self.crossover_function_weights, add_to_population=True, keep_repeats=False, mutate_until_unique=True, rng=self.rng)

        self.population.update_column(offspring, column_names="Generation", data=self.generation, )








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
                                                            n=self.evaluation_early_stop_steps)
                                        for obj_name in self.objective_names]).T

        #Get the selectors survival rates per step
        if self.selection_evaluation_early_stop is not None:
            lower = self.cur_population_size*self.selection_evaluation_early_stop[0]
            upper = self.cur_population_size*self.selection_evaluation_early_stop[1]
            #survival_counts = self.cur_population_size*(scipy.special.betainc(1,self.selection_evaluation_scaling,np.linspace(0,1,self.evaluation_early_stop_steps))*(upper-lower)+lower)

            survival_counts = np.array(beta_interpolation(start=lower, end=upper, scale=self.selection_evaluation_scaling, n=self.evaluation_early_stop_steps, n_steps=self.evaluation_early_stop_steps))
            self.survival_counts = survival_counts.astype(int)
        else:
            self.survival_counts = None



        if self.evaluation_early_stop_steps is not None:
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

        if self.max_eval_time_seconds is not None:
            theoretical_timeout = self.max_eval_time_seconds * math.ceil(len(individuals_to_evaluate) / self.n_jobs)
            theoretical_timeout = theoretical_timeout*2
        else:
            theoretical_timeout = np.inf
        scheduled_timeout_time_left = self.scheduled_timeout_time - time.time()
        parallel_timeout = min(theoretical_timeout, scheduled_timeout_time_left)
        if parallel_timeout < 0:
            parallel_timeout = 10

        #scores = tpot2.utils.eval_utils.parallel_eval_objective_list(individuals_to_evaluate, self.objective_functions, self.n_jobs, verbose=self.verbose, timeout=self.max_eval_time_seconds, budget=budget, n_expected_columns=len(self.objective_names), client=self._client, parallel_timeout=parallel_timeout, **self.objective_kwargs)
        scores, start_times, end_times, eval_errors = tpot2.utils.eval_utils.parallel_eval_objective_list2(individuals_to_evaluate, self.objective_functions, verbose=self.verbose, max_eval_time_seconds=self.max_eval_time_seconds, budget=budget, n_expected_columns=len(self.objective_names), client=self._client, scheduled_timeout_time=self.scheduled_timeout_time, **self.objective_kwargs)

        self.population.update_column(individuals_to_evaluate, column_names=self.objective_names, data=scores)
        if budget is not None:
            self.population.update_column(individuals_to_evaluate, column_names="Budget", data=budget)

        self.population.update_column(individuals_to_evaluate, column_names="Submitted Timestamp", data=start_times)
        self.population.update_column(individuals_to_evaluate, column_names="Completed Timestamp", data=end_times)
        self.population.update_column(individuals_to_evaluate, column_names="Eval Error", data=eval_errors)
        self.population.remove_invalid_from_population(column_names="Eval Error")
        self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="TIMEOUT")

    def get_unevaluated_individuals(self, column_names, budget=None, individual_list=None):
        if individual_list is not None:
            cur_pop = np.array(individual_list)
        else:
            cur_pop = np.array(self.population.population)

        if all([name_step in self.population.evaluated_individuals.columns for name_step in column_names]):
            if budget is not None:
                offspring_scores = self.population.get_column(cur_pop, column_names=column_names+["Budget"], to_numpy=False)
                #Individuals are unevaluated if we have a higher budget OR if any of the objectives are nan
                unevaluated_filter = lambda i: any(offspring_scores.loc[offspring_scores.index[i]][column_names].isna()) or (offspring_scores.loc[offspring_scores.index[i]]["Budget"] < budget)
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


        survival_selector = tpot2.selectors.survival_select_NSGA2

        ################

        objective_function_signs = np.sign(self.objective_function_weights)


        cur_individuals = self.population.population.copy()

        all_step_names = []
        for step in range(self.evaluation_early_stop_steps):
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

            if self.max_eval_time_seconds is not None:
                theoretical_timeout = self.max_eval_time_seconds * math.ceil(len(unevaluated_individuals_this_step) / self.n_jobs)
                theoretical_timeout = theoretical_timeout*2
            else:
                theoretical_timeout = np.inf
            scheduled_timeout_time_left = self.scheduled_timeout_time - time.time()
            parallel_timeout = min(theoretical_timeout, scheduled_timeout_time_left)
            if parallel_timeout < 0:
                parallel_timeout = 10

            scores, start_times, end_times, eval_errors = tpot2.utils.eval_utils.parallel_eval_objective_list2(individual_list=unevaluated_individuals_this_step,
                                    objective_list=self.objective_functions,
                                    verbose=self.verbose,
                                    max_eval_time_seconds=self.max_eval_time_seconds,
                                    step=step,
                                    budget = self.budget,
                                    generation = self.generation,
                                    n_expected_columns=len(self.objective_names),
                                    client=self._client,
                                    scheduled_timeout_time=self.scheduled_timeout_time,
                                    **self.objective_kwargs,
                                    )
            
            self.population.update_column(unevaluated_individuals_this_step, column_names=this_step_names, data=scores)
            self.population.update_column(unevaluated_individuals_this_step, column_names="Submitted Timestamp", data=start_times)
            self.population.update_column(unevaluated_individuals_this_step, column_names="Completed Timestamp", data=end_times)
            self.population.update_column(unevaluated_individuals_this_step, column_names="Eval Error", data=eval_errors)

            self.population.remove_invalid_from_population(column_names="Eval Error")
            self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="TIMEOUT")

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
            if step == self.evaluation_early_stop_steps-1:
                self.population.update_column(cur_individuals, column_names=self.objective_names, data=offspring_scores)
                if budget is not None:
                    self.population.update_column(cur_individuals, column_names="Budget", data=budget)
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
                if survival_counts is not None:
                    if step < self.evaluation_early_stop_steps - 1 and survival_counts[step]>1: #don't do selection for the last loop since they are completed
                        k = survival_counts[step] + len(invalids) #TODO can remove the min if the selections method can ignore k>population size
                        if len(cur_individuals)> 1 and k > self.n_jobs and k < len(cur_individuals):
                            weighted_scores = np.array([s * self.objective_function_weights for s in offspring_scores ])

                            new_population_index = survival_selector(weighted_scores, k=k)
                            cur_individuals = np.array(cur_individuals)[new_population_index]

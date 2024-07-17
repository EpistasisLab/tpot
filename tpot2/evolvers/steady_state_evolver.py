#All abstract methods in the Evolutionary_Optimization module
import tpot2
import typing
import tqdm
import time
import numpy as np
import os
import pickle
from tqdm.dask import TqdmCallback
import distributed
from dask.distributed import Client
from dask.distributed import LocalCluster
from tpot2.selectors import survival_select_NSGA2, tournament_selection_dominated
import math
from tpot2.utils.utils import get_thresholds, beta_interpolation, remove_items, equalize_list
import dask
import warnings


def ind_mutate(ind, rng):
    rng = np.random.default_rng(rng)
    return ind.mutate(rng=rng)

def ind_crossover(ind1, ind2, rng):
    rng = np.random.default_rng(rng)
    return ind1.crossover(ind2, rng=rng)

class SteadyStateEvolver():
    def __init__(   self,
                    individual_generator ,

                    objective_functions,
                    objective_function_weights,
                    objective_names = None,
                    objective_kwargs = None,
                    bigger_is_better = True,

                    initial_population_size = 50,
                    population_size = 50,
                    max_evaluated_individuals = None,
                    early_stop = None,
                    early_stop_seconds = None,
                    early_stop_tol = 0.001,


                    max_time_seconds=float("inf"),
                    max_eval_time_seconds=60*5,

                    n_jobs=1,
                    memory_limit="4GB",
                    client=None,


                    crossover_probability=.2,
                    mutate_probability=.7,
                    mutate_then_crossover_probability=.05,
                    crossover_then_mutate_probability=.05,
                    n_parents=2,

                    survival_selector = survival_select_NSGA2,
                    parent_selector = tournament_selection_dominated,

                    budget_range = None,
                    budget_scaling = .5,
                    individuals_until_end_budget = 1,
                    stepwise_steps = 5,


                    verbose = 0,
                    periodic_checkpoint_folder = None,
                    callback = None,

                    rng=None
                    ) -> None:

        self.rng = np.random.default_rng(rng)

        self.max_evaluated_individuals = max_evaluated_individuals
        self.individuals_until_end_budget = individuals_until_end_budget

        self.individual_generator = individual_generator
        self.population_size = population_size
        self.objective_functions = objective_functions
        self.objective_function_weights = np.array(objective_function_weights)
        self.bigger_is_better = bigger_is_better
        if not bigger_is_better:
            self.objective_function_weights = np.array(self.objective_function_weights)*-1

        self.population_size_list = None


        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.verbose  = verbose
        self.callback = callback
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

        self.initial_population_size = initial_population_size
        self.budget_range = budget_range
        self.budget_scaling = budget_scaling
        self.stepwise_steps = stepwise_steps

        self.memory_limit = memory_limit

        self.client = client


        self.survival_selector=survival_selector
        self.parent_selector=parent_selector


        total_var_p = crossover_probability + mutate_probability + mutate_then_crossover_probability + crossover_then_mutate_probability
        self.crossover_probability = crossover_probability / total_var_p
        self.mutate_probability = mutate_probability  / total_var_p
        self.mutate_then_crossover_probability= mutate_then_crossover_probability / total_var_p
        self.crossover_then_mutate_probability= crossover_then_mutate_probability / total_var_p

        self.n_parents = n_parents

        if objective_kwargs is None:
            self.objective_kwargs = {}
        else:
            self.objective_kwargs = objective_kwargs

        ###########


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
        self.early_stop_seconds = early_stop_seconds
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


        init_names = self.objective_names
        if self.budget_range is not None:
            init_names = init_names + ["Budget"]
        if self.population is None:
            self.population = tpot2.Population(column_names=init_names)
            initial_population = [next(self.individual_generator) for _ in range(self.initial_population_size)]
            self.population.add_to_population(initial_population, rng=self.rng)


    def optimize(self):

        #intialize the client
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


        self.max_queue_size = len(self._client.cluster.workers)

        #set up logging params
        evaluated_count = 0
        generations_without_improvement = np.array([0 for _ in range(len(self.objective_function_weights))])
        timestamp_of_last_improvement = np.array([time.time() for _ in range(len(self.objective_function_weights))])
        best_scores = [-np.inf for _ in range(len(self.objective_function_weights))]
        scheduled_timeout_time = time.time() + self.max_time_seconds
        budget = None

        submitted_futures = {}
        submitted_inds = set()

        start_time = time.time()

        try:


            if self.verbose >= 1:
                if self.max_evaluated_individuals is not None:
                    pbar = tqdm.tqdm(total=self.max_evaluated_individuals, miniters=1)
                else:
                    pbar = tqdm.tqdm(total=0, miniters=1)
                pbar.set_description("Evaluations")

            #submit initial population
            individuals_to_evaluate = self.get_unevaluated_individuals(self.objective_names, budget=budget,)

            for individual in individuals_to_evaluate:
                if len(submitted_futures) >= self.max_queue_size:
                    break
                future = self._client.submit(tpot2.utils.eval_utils.eval_objective_list, individual,  self.objective_functions, verbose=self.verbose, timeout=self.max_eval_time_seconds,**self.objective_kwargs)

                submitted_futures[future] = {"individual": individual,
                                            "time": time.time(),
                                            "budget": budget,}
                submitted_inds.add(individual.unique_id())
                self.population.update_column(individual, column_names="Submitted Timestamp", data=time.time())

            done = False
            start_time = time.time()

            enough_parents_evaluated=False
            while not done:

                ###############################
                # Step 1: Check for finished futures
                ###############################

                #wait for at least one future to finish or timeout
                try:
                    next(distributed.as_completed(submitted_futures, timeout=self.max_eval_time_seconds))
                except dask.distributed.TimeoutError:
                    pass
                except dask.distributed.CancelledError:
                    pass

                #Loop through all futures, collect completed and timeout futures.
                for completed_future in list(submitted_futures.keys()):
                    eval_error = None
                    #get scores and update
                    if completed_future.done(): #if future is done
                        #If the future is done but threw and error, record the error
                        if completed_future.exception() or completed_future.status == "error": #if the future is done and threw an error
                            print("Exception in future")
                            print(completed_future.exception())
                            scores = [np.nan for _ in range(len(self.objective_names))]
                            eval_error = "INVALID"
                        elif completed_future.cancelled(): #if the future is done and was cancelled
                            print("Cancelled future (likely memory related)")
                            scores = [np.nan for _ in range(len(self.objective_names))]
                            eval_error = "INVALID"
                        else: #if the future is done and did not throw an error, get the scores
                            try:
                                scores = completed_future.result()

                                #check if scores contain "INVALID" or "TIMEOUT"
                                if "INVALID" in scores:
                                    eval_error = "INVALID"
                                    scores = [np.nan]
                                elif "TIMEOUT" in scores:
                                    eval_error = "TIMEOUT"
                                    scores = [np.nan]

                            except Exception as e:
                                print("Exception in future, but not caught by dask")
                                print(e)
                                print(completed_future.exception())
                                print(completed_future)
                                print("status", completed_future.status)
                                print("done", completed_future.done())
                                print("cancelld ", completed_future.cancelled())
                                scores = [np.nan for _ in range(len(self.objective_names))]
                                eval_error = "INVALID"
                    else: #if future is not done

                        if self.max_eval_time_seconds is not None:
                            #check if the future has been running for too long, cancel the future
                            if time.time() - submitted_futures[completed_future]["time"] > self.max_eval_time_seconds*1.25:
                                completed_future.cancel()

                                if self.verbose >= 4:
                                    print(f'WARNING AN INDIVIDUAL TIMED OUT (Fallback): \n {submitted_futures[completed_future]} \n')

                                scores = [np.nan for _ in range(len(self.objective_names))]
                                eval_error = "TIMEOUT"
                            else:
                                continue #otherwise, continue to next future



                    #update population
                    this_individual = submitted_futures[completed_future]["individual"]
                    this_budget = submitted_futures[completed_future]["budget"]
                    this_time = submitted_futures[completed_future]["time"]

                    if len(scores) < len(self.objective_names):
                        scores = [scores[0] for _ in range(len(self.objective_names))]
                    self.population.update_column(this_individual, column_names=self.objective_names, data=scores)
                    self.population.update_column(this_individual, column_names="Completed Timestamp", data=time.time())
                    self.population.update_column(this_individual, column_names="Eval Error", data=eval_error)
                    if budget is not None:
                        self.population.update_column(this_individual, column_names="Budget", data=this_budget)

                    submitted_futures.pop(completed_future)
                    submitted_inds.add(this_individual.unique_id())
                    if self.verbose >= 1:
                        pbar.update(1)

                #now we have a list of completed futures

                self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="INVALID")
                self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="TIMEOUT")


                ###############################
                # Step 2: Early Stopping
                ###############################
                if self.verbose >= 3:
                    sign = np.sign(self.objective_function_weights)
                    valid_df = self.population.evaluated_individuals[~self.population.evaluated_individuals[["Eval Error"]].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names]*sign
                    cur_best_scores = valid_df.max(axis=0)*sign
                    cur_best_scores = cur_best_scores.to_numpy()
                    for i, obj in enumerate(self.objective_names):
                        print(f"Best {obj} score: {cur_best_scores[i]}")

                if self.early_stop or self.early_stop_seconds:
                    if self.budget is None or self.budget>=self.budget_range[-1]: #self.budget>=1:
                        #get sign of objective_function_weights
                        sign = np.sign(self.objective_function_weights)
                        #get best score for each objective
                        valid_df = self.population.evaluated_individuals[~self.population.evaluated_individuals[["Eval Error"]].isin(["TIMEOUT","INVALID"]).any(axis=1)][self.objective_names]*sign
                        cur_best_scores = valid_df.max(axis=0)
                        cur_best_scores = cur_best_scores.to_numpy()
                        #cur_best_scores =  self.population.get_column(self.population.population, column_names=self.objective_names).max(axis=0)*sign #TODO this assumes the current population is the best

                        improved = ( np.array(cur_best_scores) - np.array(best_scores) >= np.array(self.early_stop_tol) )
                        not_improved = np.logical_not(improved)
                        generations_without_improvement = generations_without_improvement * not_improved + not_improved #set to zero if not improved, else increment

                        timestamp_of_last_improvement = timestamp_of_last_improvement * not_improved + time.time()*improved #set to current time if improved

                        pass
                        #update best score
                        best_scores = [max(best_scores[i], cur_best_scores[i]) for i in range(len(self.objective_names))]

                        if self.early_stop:
                            if all(generations_without_improvement>self.early_stop):
                                if self.verbose >= 3:
                                    print(f"Early stop ({self.early_stop} individuals evaluated without improvement)")
                                break

                        if self.early_stop_seconds:
                            if any(time.time() - timestamp_of_last_improvement > self.early_stop_seconds):
                                if self.verbose >= 3:
                                    print(f"Early stop  ({self.early_stop_seconds} seconds passed without improvement)")
                                break

                #if we evaluated enough individuals or time is up, stop
                if self.max_time_seconds is not None and time.time() - start_time > self.max_time_seconds:
                    if self.verbose >= 3:
                        print("Time limit reached")
                    done = True
                    break

                if self.max_evaluated_individuals is not None and len(self.population.evaluated_individuals.dropna(subset=self.objective_names)) >= self.max_evaluated_individuals:
                    print("Evaluated enough individuals")
                    done = True
                    break

                ###############################
                # Step 3: Submit unevaluated individuals from the initial population
                ###############################
                individuals_to_evaluate = self.get_unevaluated_individuals(self.objective_names, budget=budget,)
                individuals_to_evaluate = [ind for ind in individuals_to_evaluate if ind.unique_id() not in submitted_inds]
                for individual in individuals_to_evaluate:
                    if self.max_queue_size > len(submitted_futures):
                        future = self._client.submit(tpot2.utils.eval_utils.eval_objective_list, individual,  self.objective_functions, verbose=self.verbose, timeout=self.max_eval_time_seconds,**self.objective_kwargs)

                        submitted_futures[future] = {"individual": individual,
                                                    "time": time.time(),
                                                    "budget": budget,}
                        submitted_inds.add(individual.unique_id())

                        self.population.update_column(individual, column_names="Submitted Timestamp", data=time.time())


                ###############################
                # Step 4: Survival Selection
                ###############################
                if self.survival_selector is not None:
                    parents_df = self.population.get_column(self.population.population, column_names=self.objective_names + ["Individual"], to_numpy=False)
                    evaluated = parents_df[~parents_df[self.objective_names].isna().any(axis=1)]
                    if len(evaluated) > self.population_size:
                        unevaluated = parents_df[parents_df[self.objective_names].isna().any(axis=1)]

                        cur_evaluated_population = parents_df["Individual"].to_numpy()
                        if len(cur_evaluated_population) > self.population_size:
                            scores = evaluated[self.objective_names].to_numpy()
                            weighted_scores = scores * self.objective_function_weights
                            new_population_index = np.ravel(self.survival_selector(weighted_scores, k=self.population_size, rng=self.rng)) #TODO make it clear that we are concatenating scores...

                            #set new population
                            try:
                                cur_evaluated_population = np.array(cur_evaluated_population)[new_population_index]
                                cur_evaluated_population = np.concatenate([cur_evaluated_population, unevaluated["Individual"].to_numpy()])
                                self.population.set_population(cur_evaluated_population, rng=self.rng)
                            except Exception as e:
                                print("Exception in survival selection")
                                print(e)
                                print("new_population_index", new_population_index)
                                print("cur_evaluated_population", cur_evaluated_population)
                                print("unevaluated", unevaluated)
                                print("evaluated", evaluated)
                                print("scores", scores)
                                print("weighted_scores", weighted_scores)
                                print("self.objective_function_weights", self.objective_function_weights)
                                print("self.population_size", self.population_size)
                                print("parents_df", parents_df)

                ###############################
                # Step 5: Parent Selection and Variation
                ###############################
                n_individuals_to_submit = self.max_queue_size - len(submitted_futures)
                if n_individuals_to_submit > 0:
                    #count non-nan values in the objective columns
                    if not enough_parents_evaluated:
                        parents_df = self.population.get_column(self.population.population, column_names=self.objective_names, to_numpy=False)
                        scores = parents_df[self.objective_names[0]].to_numpy()
                        #count non-nan values in the objective columns
                        n_evaluated = np.count_nonzero(~np.isnan(scores))
                        if n_evaluated >0 :
                            enough_parents_evaluated=True
                    
                    # parents_df = self.population.get_column(self.population.population, column_names=self.objective_names+ ["Individual"], to_numpy=False)
                    # parents_df = parents_df[~parents_df[self.objective_names].isin(["TIMEOUT","INVALID"]).any(axis=1)]
                    # parents_df = parents_df[~parents_df[self.objective_names].isna().any(axis=1)]

                    # cur_evaluated_population = parents_df["Individual"].to_numpy()
                    # if len(cur_evaluated_population) > 0:
                    #     scores = parents_df[self.objective_names].to_numpy()
                    #     weighted_scores = scores * self.objective_function_weights
                    #     #number of crossover pairs and mutation only parent to generate

                    #     if len(parents_df) < 2:
                    #         var_ops = ["mutate" for _ in range(n_individuals_to_submit)]
                    #     else:
                    #         var_ops = [self.rng.choice(["crossover","mutate_then_crossover","crossover_then_mutate",'mutate'],p=[self.crossover_probability,self.mutate_then_crossover_probability, self.crossover_then_mutate_probability,self.mutate_probability]) for _ in range(n_individuals_to_submit)]

                    #     parents = []
                    #     for op in var_ops:
                    #         if op == "mutate":
                    #             parents.extend(np.array(cur_evaluated_population)[self.parent_selector(weighted_scores, k=1, n_parents=1, rng=self.rng)])
                    #         else:
                    #             parents.extend(np.array(cur_evaluated_population)[self.parent_selector(weighted_scores, k=1, n_parents=2, rng=self.rng)])

                    #     #_offspring = self.population.create_offspring2(parents, var_ops, rng=self.rng, add_to_population=True)
                    #     offspring = self.population.create_offspring2(parents, var_ops, [ind_mutate], None, [ind_crossover], None, add_to_population=True, keep_repeats=False, mutate_until_unique=True, rng=self.rng)

                    if enough_parents_evaluated:

                        parents = self.population.parent_select(selector=self.parent_selector, weights=self.objective_function_weights, columns_names=self.objective_names, k=n_individuals_to_submit, n_parents=2, rng=self.rng)
                        p = np.array([self.crossover_probability, self.mutate_then_crossover_probability, self.crossover_then_mutate_probability, self.mutate_probability])
                        p = p / p.sum()
                        var_op_list = self.rng.choice(["crossover", "mutate_then_crossover", "crossover_then_mutate", "mutate"], size=n_individuals_to_submit, p=p)

                        for i, op in enumerate(var_op_list):
                            if op == "mutate":
                                parents[i] = parents[i][0] #mutations take a single individual

                        offspring = self.population.create_offspring2(parents, var_op_list, [ind_mutate], None, [ind_crossover], None, add_to_population=True, keep_repeats=False, mutate_until_unique=True, rng=self.rng)

                    # If we don't have enough evaluated individuals to use as parents for variation, we create new individuals randomly
                    # This can happen if the individuals in the initial population are invalid
                    elif len(submitted_futures) < self.max_queue_size:

                        initial_population = self.population.evaluated_individuals.iloc[:self.initial_population_size*3]
                        invalid_initial_population = initial_population[initial_population[["Eval Error"]].isin(["TIMEOUT","INVALID"]).any(axis=1)]
                        if len(invalid_initial_population) >= self.initial_population_size*3: #if all individuals in the 3*initial population are invalid
                            raise Exception("No individuals could be evaluated in the initial population. This may indicate a bug in the configuration, included models, or objective functions. Set verbose>=4 to see the errors that caused individuals to fail.")

                        n_individuals_to_create = self.max_queue_size - len(submitted_futures)
                        initial_population = [next(self.individual_generator) for _ in range(n_individuals_to_create)]
                        self.population.add_to_population(initial_population)




                ###############################
                # Step 6: Add Unevaluated Individuals Generated by Variation
                ###############################
                individuals_to_evaluate = self.get_unevaluated_individuals(self.objective_names, budget=budget,)
                individuals_to_evaluate = [ind for ind in individuals_to_evaluate if ind.unique_id() not in submitted_inds]
                for individual in individuals_to_evaluate:
                    if self.max_queue_size > len(submitted_futures):
                        future = self._client.submit(tpot2.utils.eval_utils.eval_objective_list, individual,  self.objective_functions, verbose=self.verbose, timeout=self.max_eval_time_seconds,**self.objective_kwargs)

                        submitted_futures[future] = {"individual": individual,
                                                    "time": time.time(),
                                                    "budget": budget,}
                        submitted_inds.add(individual.unique_id())
                        self.population.update_column(individual, column_names="Submitted Timestamp", data=time.time())


                #Checkpointing
                if self.population_file is not None: # and time.time() - last_save_time > 60*10:
                    pickle.dump(self.population, open(self.population_file, "wb"))



        except KeyboardInterrupt:
            if self.verbose >= 3:
                print("KeyboardInterrupt")

        ###############################
        # Step 7: Cleanup
        ###############################

        self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="INVALID")
        self.population.remove_invalid_from_population(column_names="Eval Error", invalid_value="TIMEOUT")


        #done, cleanup futures
        for future in submitted_futures.keys():
            future.cancel()

        #checkpoint
        if self.population_file is not None:
            pickle.dump(self.population, open(self.population_file, "wb"))

        if self.client is None: #If we created our own client, close it
            self._client.close()
            self._cluster.close()

        tpot2.utils.get_pareto_frontier(self.population.evaluated_individuals, column_names=self.objective_names, weights=self.objective_function_weights)



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

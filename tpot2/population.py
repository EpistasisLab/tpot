import numpy as np
import copy
import copy
import typing
import tpot2
from tpot2.individual import BaseIndividual
from traitlets import Bool
import collections
import pandas as pd
from joblib import Parallel, delayed
import copy
import pickle
import dask

def mutate(individual):
    if isinstance(individual, collections.abc.Iterable):
        for ind in individual:
            ind.mutate()
    else:
        individual.mutate()
    return individual

def crossover(parents):
    parents[0].crossover(parents[1])
    return parents[0]

def mutate_and_crossover(parents):
    parents[0].crossover(parents[1])
    parents[0].mutate()
    parents[1].mutate()
    return parents

def crossover_and_mutate(parents):
    for p in parents:
        p.mutate()
    parents[0].crossover(parents[1])
    return parents[0]


built_in_var_ops_dict = {"mutate":mutate, 
                        "crossover":crossover, 
                        "mutate_then_crossover":mutate_and_crossover, 
                        "crossover_then_mutate":crossover_and_mutate}



    
class Population():
    '''
    Primary usage is to keep track of evaluated individuals
    
    Parameters
    ----------
    initial_population : {list of BaseIndividuals}, default=None
        Initial population to start with. If None, start with an empty population.
    use_unique_id : {Bool}, default=True
        If True, individuals are treated as unique if they have the same unique_id().
        If False, all new individuals are treated as unique.
    callback : {function}, default=None
        NOT YET IMPLEMENTED
        A function to call after each generation. The function should take a Population object as its only argument.
    
    Attributes
    ----------
    population : {list of BaseIndividuals}
        The current population of individuals. Contains the live instances of BaseIndividuals.
    evaluated_individuals : {dict}
        A dictionary of dictionaries. The keys are the unique_id() or self of each BaseIndividual.
        Can be thought of as a table with the unique_id() as the row index and the inner dictionary keys as the columns.
    '''
    def __init__(   self,
                    column_names: typing.List[str] = None,
                    n_jobs: int = 1,
                    callback=None,
                    ) -> None:

        if column_names is not None:
            
            column_names = column_names+["Parents", "Variation_Function"]
        else:
            column_names = ["Parents", "Variation_Function"]
        self.evaluated_individuals = pd.DataFrame(columns=column_names)
        self.evaluated_individuals["Parents"] = self.evaluated_individuals["Parents"].astype('object')
        self.use_unique_id = True #Todo clean this up. perhaps pull unique_id() out of baseestimator and have it be supplied as a function
        self.n_jobs = n_jobs
        self.callback=callback
        self.population = []



    #remove individuals that either do not have a column_name value or a nan in that value
    #TODO take into account when the value is not a list/tuple?
    #TODO make invalid a global variable?
    def remove_invalid_from_population(self, column_names, invalid_value = "INVALID"):
        '''
        Remove individuals from the live population if either do not have a value in the column_name column or if the value contains np.nan.
        
        Parameters
        ----------
        column_name : {str}
            The name of the column to check for np.nan values.
        
        Returns
        -------
        None
        '''
        if isinstance(column_names, str): #TODO check this
            column_names = [column_names]
        new_pop = []
        is_valid = lambda ind: ind.unique_id() not in self.evaluated_individuals.index or invalid_value not in self.evaluated_individuals.loc[ind.unique_id(),column_names].to_list()
        self.population = [ind for ind in self.population if is_valid(ind)]

        

    # takes the list of individuals and adds it to the live population list. 
    # if keep_repeats is False, repeated individuals are not added to the population
    # returns a list of individuals added to the live population  
    #TODO make keep repeats allow for previously evaluated individuals,
    #but make sure that the live population only includes one of each, no repeats
    def add_to_population(self, individuals: typing.List[BaseIndividual], keep_repeats=False, mutate_until_unique=True):
        '''
        Add individuals to the live population. Add individuals to the evaluated_individuals if they are not already there.
        
        Parameters:
        -----------
        individuals : {list of BaseIndividuals}
            The individuals to add to the live population.
        keep_repeats : {bool}, default=False
            If True, allow the population to have repeated individuals.
            If False, only add individuals that have not yet been added to geneology.
        '''
        if not isinstance(individuals, collections.abc.Iterable):
            individuals = [individuals]

        new_individuals = []
        #TODO check for proper inputs
        for individual in individuals:
            key = individual.unique_id()

            if key not in self.evaluated_individuals.index: #If its new, we always add it
                self.evaluated_individuals.loc[key] = np.nan
                self.evaluated_individuals.loc[key,"Individual"] = copy.deepcopy(individual)
                self.population.append(individual)
                new_individuals.append(individual)

            else:#If its old
                if keep_repeats: #If we want to keep repeats, we add it
                    self.population.append(individual)
                    new_individuals.append(individual)
                elif mutate_until_unique: #If its old and we don't want repeats, we can optionally mutate it until it is unique
                    for _ in range(20):
                        individual = copy.deepcopy(individual)
                        individual.mutate()
                        key = individual.unique_id()
                        if key not in self.evaluated_individuals.index:
                            self.evaluated_individuals.loc[key] = np.nan
                            self.evaluated_individuals.loc[key,"Individual"] = copy.deepcopy(individual)
                            self.population.append(individual)
                            new_individuals.append(individual)
                            break
                    
        return new_individuals


    def update_column(self, individual, column_names, data):
        '''
        Update the column_name column in the evaluated_individuals with the data.
        If the data is a list, it must be the same length as the evaluated_individuals.
        If the data is a single value, it will be applied to all individuals in the evaluated_individuals.
        '''
        if isinstance(individual, collections.abc.Iterable):
            if self.use_unique_id:
                key = [ind.unique_id() for ind in individual]
            else:
                key = individual
        else:
            if self.use_unique_id:
                key = individual.unique_id()
            else:
                key = individual

        self.evaluated_individuals.loc[key,column_names] = data

    
    def get_column(self, individual, column_names=None, to_numpy=True):
        '''
        Update the column_name column in the evaluated_individuals with the data.
        If the data is a list, it must be the same length as the evaluated_individuals.
        If the data is a single value, it will be applied to all individuals in the evaluated_individuals.
        '''
        if isinstance(individual, collections.abc.Iterable):
            if self.use_unique_id:
                key = [ind.unique_id() for ind in individual]
            else:
                key = individual
        else:
            if self.use_unique_id:
                key = individual.unique_id()
            else:
                key = individual

        if column_names is not None:
            slice = self.evaluated_individuals.loc[key,column_names]
        else:
            slice = self.evaluated_individuals.loc[key]
        if to_numpy:
            slice.reset_index(drop=True, inplace=True)
            return slice.to_numpy()
        else:
            return slice


    #returns the individuals without a 'column' as a key in geneology
    #TODO make sure not to get repeats in this list even if repeats are in the "live" population
    def get_unevaluated_individuals(self, column_names, individual_list=None):
        if individual_list is None:
            individual_list = self.population
        
        if self.use_unique_id:
            unevaluated_filter = lambda individual: individual.unique_id() not in self.evaluated_individuals.index or any(self.evaluated_individuals.loc[individual.unique_id(), column_names].isna())
        else:
            unevaluated_filter = lambda individual: individual not in self.evaluated_individuals.index or any(self.evaluated_individuals.loc[individual.unique_id(), column_names].isna())
        
        return [individual for individual in individual_list if unevaluated_filter(individual)]    

    # def get_valid_evaluated_individuals_df(self, column_names_to_check, invalid_values=["TIMEOUT","INVALID"]):
    #     '''
    #     Returns a dataframe of the evaluated individuals that do no have invalid_values in column_names_to_check.
    #     '''
    #     return self.evaluated_individuals[~self.evaluated_individuals[column_names_to_check].isin(invalid_values).any(axis=1)]

    #the live population empied and is set to new_population
    def set_population(self,  new_population, keep_repeats=True):
        '''
        sets population to new population
        for selection?
        '''
        self.population = []
        self.add_to_population(new_population, keep_repeats=keep_repeats)

    #TODO should we just generate one offspring per crossover? 
    def create_offspring(self, parents_list, var_op_list, add_to_population=True, keep_repeats=False, mutate_until_unique=True, n_jobs=1):
        '''
        parents_list: a list of lists of parents. 
        var_op_list: a list of var_ops to apply to each list of parents. Should be the same length as parents_list.

        for example:
        parents_list = [[parent1, parent2], [parent3]]
        var_op_list = ["crossover", "mutate"]

        This will apply crossover to parent1 and parent2 and mutate to parent3.

        Creates offspring from parents using the var_op_list.
        If string, will use a built in method 
            - "crossover" : crossover
            - "mutate" : mutate
            - "mutate_and_crossover" : mutate_and_crossover
            - "cross_and_mutate" : cross_and_mutate
        '''
        new_offspring = []
        all_offspring = parallel_create_offspring(parents_list, var_op_list, n_jobs=n_jobs)

        for parents, offspring, var_op in zip(parents_list, all_offspring, var_op_list):
            
            # if var_op in built_in_var_ops_dict:
            #     var_op = built_in_var_ops_dict[var_op]

            # offspring = copy.deepcopy(parents)
            # offspring = var_op(offspring)
            # if isinstance(offspring, collections.abc.Iterable):
            #     offspring = offspring[0] 

            if add_to_population:
                added = self.add_to_population(offspring, keep_repeats=keep_repeats, mutate_until_unique=mutate_until_unique)
                if len(added) > 0:
                    for new_child in added:
                        parent_keys = [parent.unique_id() for parent in parents]
                        if not pd.api.types.is_object_dtype(self.evaluated_individuals["Parents"]): #TODO Is there a cleaner way of doing this? Not required for some python environments?
                            self.evaluated_individuals["Parents"] = self.evaluated_individuals["Parents"].astype('object')
                        self.evaluated_individuals.at[new_child.unique_id(),"Parents"] = tuple(parent_keys)
                        
                        #if var_op is a function
                        if hasattr(var_op, '__call__'):
                            self.evaluated_individuals.at[new_child.unique_id(),"Variation_Function"] = var_op.__name__
                        else:
                            self.evaluated_individuals.at[new_child.unique_id(),"Variation_Function"] = var_op
                        
                        
                        new_offspring.append(new_child)

            else:
                new_offspring.append(offspring)
                            
            
        return new_offspring

   

def get_id(individual):
    return individual.unique_id()

def parallel_create_offspring(parents_list, var_op_list, n_jobs=1):
    delayed_offspring = []
    for parents, var_op in zip(parents_list,var_op_list):
        #TODO put this loop in population class
        if var_op in built_in_var_ops_dict:
            var_op = built_in_var_ops_dict[var_op]
        delayed_offspring.append(dask.delayed(copy_and_change)(parents, var_op))

    offspring = dask.compute(*delayed_offspring,
                             num_workers=n_jobs, threads_per_worker=1)
    return offspring

def copy_and_change(parents, var_op):
    offspring = copy.deepcopy(parents)
    offspring = var_op(offspring)
    if isinstance(offspring, collections.abc.Iterable):
        offspring = offspring[0]
    return offspring

def parallel_get_id(n_jobs, individual_list):
    id_list = Parallel(n_jobs=n_jobs)(delayed(get_id)(ind)  for ind in individual_list)
    return id_list

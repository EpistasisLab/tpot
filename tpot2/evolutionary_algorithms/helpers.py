"""
Some helper functions for EA

"""
import random
import copy
import numpy as np

def applyMutation(individual_list):
    individual, = random.choices(individual_list, k=1)
    individual = copy.deepcopy(individual)
    individual.mutate()
    return individual

def applyCrossover(individual_list):
    individual_1, individual_2 = random.choices(individual_list, k=2)
    individual_1 = copy.deepcopy(individual_1)
    individual_2 = copy.deepcopy(individual_2)
    individual_1.crossover(individual_2)
    return individual_1, individual_2


def applyVariationOperators(individual_list, number_of_children_to_produce, variation_probabilities):
    offspring = []
    count = 0 
    while count < number_of_children_to_produce:
        op_choice, = random.choices(list(variation_probabilities.keys()), k=1, weights=list(variation_probabilities.values()))

        if op_choice=='mutation':
            individual_1 = applyMutation(individual_list)
            offspring.append(individual_1)
            count += 1

        elif op_choice=='crossover':
            individual_1, individual_2 = applyCrossover(individual_list)
            offspring.append(individual_1)
            offspring.append(individual_2)
            count += 2

        elif op_choice=='mutation_then_crossover':
            individual_1 = applyMutation(individual_list)
            individual_2 = applyMutation(individual_list)
            individual_1.crossover(individual_2)
            offspring.append(individual_1)
            offspring.append(individual_2)
            count += 2

        elif op_choice=='crossover_then_mutation':
            iindividual_1, individual_2 = applyCrossover(individual_list)
            individual_1.mutate()
            individual_2.mutate()
            offspring.append(individual_1)
            offspring.append(individual_2)
            count += 2

    return offspring[0:number_of_children_to_produce]


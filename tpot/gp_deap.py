# -*- coding: utf-8 -*-

"""
Copyright 2015-Present Randal S. Olson

This file is part of the TPOT library.

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

Most of the code in this file was copied from the DEAP project, which
can be found on GitHub at https://github.com/DEAP/deap. We copied these
functions into TPOT to customize or fix portions of the code for our
own purposes in TPOT.

"""

import numpy as np
from deap import tools, gp
from inspect import isclass

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The variation goes as follow. On each of the *lambda_* iteration, it
    selects one of the three operations; crossover, mutation or reproduction.
    In the case of a crossover, two individuals are selected at random from
    the parental population :math:`P_\mathrm{p}`, those individuals are cloned
    using the :meth:`toolbox.clone` method and then mated using the
    :meth:`toolbox.mate` method. Only the first child is appended to the
    offspring population :math:`P_\mathrm{o}`, the second child is discarded.
    In the case of a mutation, one individual is selected at random from
    :math:`P_\mathrm{p}`, it is cloned and then mutated using using the
    :meth:`toolbox.mutate` method. The resulting mutant is appended to
    :math:`P_\mathrm{o}`. In the case of a reproduction, one individual is
    selected at random from :math:`P_\mathrm{p}`, cloned and appended to
    :math:`P_\mathrm{o}`.
    This variation is named *Or* beceause an offspring will never result from
    both operations crossover and mutation. The sum of both probabilities
    shall be in :math:`[0, 1]`, the reproduction probability is
    1 - *cxpb* - *mutpb*.
    """
    offspring = []
    for _ in range(lambda_):
        op_choice = np.random.random()
        if op_choice < cxpb:            # Apply crossover
            idxs = np.random.randint(0, len(population),size=2)
            ind1, ind2 = toolbox.clone(population[idxs[0]]), toolbox.clone(population[idxs[1]])
            ind_str = str(ind1)
            num_loop = 0
            while ind_str == str(ind1) and num_loop < 50 : # 50 loops at most to generate a different individual by crossover
                ind1, ind2 = toolbox.mate(ind1, ind2)
                num_loop += 1
            if ind_str != str(ind1): # check if crossover happened
                del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            idx = np.random.randint(0, len(population))
            ind = toolbox.clone(population[idx])
            ind_str = str(ind)
            num_loop = 0
            while ind_str == str(ind) and num_loop < 50 : # 50 loops at most to generate a different individual by mutation
                ind, = toolbox.mutate(ind)
                num_loop += 1
            if ind_str != str(ind): # check if mutation happened
                del ind.fitness.values
            offspring.append(ind)
        else: # Apply reproduction
            idx = np.random.randint(0, len(population))
            offspring.append(toolbox.clone(population[idx]))

    return offspring

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar,
                   stats=None, halloffame=None, verbose=0, max_time_mins = None):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param pbar: processing bar
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    The algorithm takes in a population and evolves it in place using the
    :func:`varOr` function. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varOr` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
            evaluate(offspring)
            population = select(population + offspring, mu)
    First, the individuals having an invalid fitness are evaluated. Second,
    the evolutionary loop begins by producing *lambda_* offspring from the
    population, the offspring are generated by the :func:`varOr` function. The
    offspring are then evaluated and the next generation population is
    selected from both the offspring **and** the population. Finally, when
    *ngen* generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox. This algorithm uses the :func:`varOr`
    variation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # update pbar for valid_ind
        if not pbar.disable:
            pbar.update(len(offspring)-len(invalid_ind))
            if not (max_time_mins is None) and pbar.n >= pbar.total:
                pbar.total += lambda_

        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # pbar process
        if not pbar.disable:
            # Print only the best individual fitness
            if verbose == 2:
                high_score = abs(max([halloffame.keys[x].wvalues[1] for x in range(len(halloffame.keys))]))
                pbar.write('Generation {0} - Current best internal CV score: {1}'.format(gen, high_score))

            # Print the entire Pareto front
            elif verbose == 3:
                pbar.write('Generation {} - Current Pareto front scores:'.format(gen))
                for pipeline, pipeline_scores in zip(halloffame.items, reversed(halloffame.keys)):
                    pbar.write('{}\t{}\t{}'.format(int(abs(pipeline_scores.wvalues[0])),
                                                         abs(pipeline_scores.wvalues[1]),
                                                         pipeline))
                pbar.write('')

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    return population, logbook


# point mutation function
def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive no matter if it has the same number of arguments from the :attr:`pset`
    attribute of the individual.
    Parameters
    ----------
    individual: DEAP individual
        A list of pipeline operators and model parameters that can be
        compiled by DEAP into a callable function

    Returns
    -------
    individual: DEAP individual
        Returns the individual with one of point mutation applied to it

    """

    index = np.random.randint(0, len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)

    if node.arity == 0:  # Terminal
        term = np.random.choice(pset.terminals[node.ret])
        if isclass(term):
            term = term()
        individual[index] = term
    else:   # Primitive
        # find next primitive if any
        rindex = None
        if index + 1 < len(individual):
            for i, tmpnode in enumerate(individual[index+1:], index+ 1):
                if isinstance(tmpnode, gp.Primitive) and tmpnode.ret in tmpnode.args:
                    rindex = i
        #pset.primitives[node.ret] can get a list of the type of node
        # for example: if op.root is True then the node.ret is Output_DF object
        # based on the function _setup_pset. Then primitives is the list of classifor or regressor
        primitives = pset.primitives[node.ret]
        if len(primitives) != 0:
            new_node = np.random.choice(primitives)
            new_subtree = [None] * len(new_node.args)
            if rindex:
                rnode = individual[rindex]
                rslice = individual.searchSubtree(rindex)
                # find position for passing return values to next operator
                position = np.random.choice([i for i, a in enumerate(new_node.args) if a == rnode.ret])
            else:
                position = None
            for i, arg_type in enumerate(new_node.args):
                if i != position:
                    term = np.random.choice(pset.terminals[arg_type])
                    if isclass(term):
                        term = term()
                    new_subtree[i] = term
            # paste the subtree to new node
            if rindex:
                new_subtree[position:position + 1] = individual[rslice]
            # combine with primitives
            new_subtree.insert(0, new_node)
            individual[slice_] = new_subtree
    return individual,

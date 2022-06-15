# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

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

"""

from tpot import TPOTClassifier
from tpot.gp_deap import initialize_stats_dict

from deap import creator

import random


def test_dict_initialization():
    """Asserts that gp_deap.initialize_stats_dict initializes individual statistics correctly"""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    tb = tpot_obj._toolbox

    test_ind = tb.individual()
    initialize_stats_dict(test_ind)

    assert test_ind.statistics['generation'] == 0
    assert test_ind.statistics['crossover_count'] == 0
    assert test_ind.statistics['mutation_count'] == 0
    assert test_ind.statistics['predecessor'] == ('ROOT',)


def test_mate_operator_stats_update():
    """Assert that self._mate_operator updates stats as expected."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    ind1 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )
    ind2 = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=True),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=2, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )

    initialize_stats_dict(ind1)
    initialize_stats_dict(ind2)

    # Randomly mutate the statistics
    ind1.statistics["crossover_count"] = random.randint(0, 10)
    ind1.statistics["mutation_count"] = random.randint(0, 10)
    ind2.statistics["crossover_count"] = random.randint(0, 10)
    ind2.statistics["mutation_count"] = random.randint(0, 10)

    # set as evaluated pipelines in tpot_obj.evaluated_individuals_
    tpot_obj.evaluated_individuals_[str(ind1)] = tpot_obj._combine_individual_stats(2, 0.99, ind1.statistics)
    tpot_obj.evaluated_individuals_[str(ind2)] = tpot_obj._combine_individual_stats(2, 0.99, ind2.statistics)

    # Doing 10 tests
    for _ in range(10):
        offspring1, _ = tpot_obj._mate_operator(ind1, ind2)

        assert offspring1.statistics['crossover_count'] == ind1.statistics['crossover_count'] + ind2.statistics['crossover_count'] + 1
        assert offspring1.statistics['mutation_count'] == ind1.statistics['mutation_count'] + ind2.statistics['mutation_count']
        assert offspring1.statistics['predecessor'] == (str(ind1), str(ind2))

        # Offspring replaces on of the two predecessors
        # Don't need to worry about cloning
        if random.random() < 0.5:
            ind1 = offspring1
        else:
            ind2 = offspring1


def test_mut_operator_stats_update():
    """Asserts that self._random_mutation_operator updates stats as expected."""
    tpot_obj = TPOTClassifier()
    tpot_obj._fit_init()
    ind = creator.Individual.from_string(
        'KNeighborsClassifier('
        'BernoulliNB(input_matrix, BernoulliNB__alpha=10.0, BernoulliNB__fit_prior=False),'
        'KNeighborsClassifier__n_neighbors=10, '
        'KNeighborsClassifier__p=1, '
        'KNeighborsClassifier__weights=uniform'
        ')',
        tpot_obj._pset
    )

    initialize_stats_dict(ind)

    ind.statistics["crossover_count"] = random.randint(0, 10)
    ind.statistics["mutation_count"] = random.randint(0, 10)

    # set as evaluated pipelines in tpot_obj.evaluated_individuals_
    tpot_obj.evaluated_individuals_[str(ind)] = tpot_obj._combine_individual_stats(2, 0.99, ind.statistics)

    for _ in range(10):
        offspring, = tpot_obj._random_mutation_operator(ind)
        
        assert offspring.statistics['crossover_count'] == ind.statistics['crossover_count']
        assert offspring.statistics['mutation_count'] == ind.statistics['mutation_count'] + 1
        assert offspring.statistics['predecessor'] == (str(ind),)

        ind = offspring

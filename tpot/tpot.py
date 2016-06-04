# -*- coding: utf-8 -*-

"""
Copyright 2016 Randal S. Olson
This file is part of the TPOT library.
The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.
The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the TPOT library. If not, see http://www.gnu.org/licenses/.
"""

from .models.base import (
    PipelineEstimator,
)
from .models.cluster import (
    feat_agg,
)
from .models.decomposition import (
    fast_ica,
)
from .models.ensemble import (
    ada_boost,
    extra_trees,
    gradient_boost,
    random_forest,
)
from .models.feature_selection import (
    rfe,
    select_kbest,
    select_percentile,
    variance_threshold,
)
from .models.linear_model import (
    logistic_regression,
    passive_aggressive,
)
from .models.neighbors import (
    knnc,
)
from .models.svm import (
    # svc,
    linear_svc,
)
from .models.tree import (
    decision_tree,
)

from .primitives import (
    get_fitness_attr,
    _div,
    _combined_selection_operator,
    _combine_dfs,
    _zero_count,
)

from pandas import np
from sklearn.cross_validation import (
    train_test_split,
)
from pandas import (
    DataFrame,
    MultiIndex,
)
from sklearn.base import (
    ClassifierMixin,
    BaseEstimator,
)
from random import (
    random,
)
from deap import (
    algorithms,
    tools,
    gp,
    base,
    creator,
)
from toolz import (
    partial,
)

import operator

default_class_column = 'species'


def pareto_eq(a, b):
    """Similarity function for the hall fof famne
    """
    return np.all(get_fitness_attr(a) == get_fitness_attr(b))


class DeapSetup(object):
    base_models = [
        fast_ica,
        ada_boost, extra_trees, gradient_boost, random_forest,
        logistic_regression, passive_aggressive,
        decision_tree,
        linear_svc,
        #svc,
        feat_agg,
        knnc,
        variance_threshold, select_kbest, select_percentile, rfe,
    ]
    # This can be changed for numpy arrays later
    input_types = [DataFrame]

    pset = gp.PrimitiveSetTyped('MAIN', [DataFrame], DataFrame)
    pset.addPrimitive(operator.add, [int, int], int)
    pset.addPrimitive(operator.sub, [int, int], int)
    pset.addPrimitive(operator.mul, [int, int], int)
    pset.addPrimitive(
        _combine_dfs, [DataFrame, DataFrame], DataFrame, name='_combine_dfs'
    )
    pset.addPrimitive(_zero_count, [DataFrame], DataFrame, name='_zero_count')
    pset.addPrimitive(_div, [int, int], float, name='_div')

    for val in range(0, 101):
        pset.addTerminal(val, int)
    for val in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
        pset.addTerminal(val, float)

    creator.create('FitnessMulti', base.Fitness, weights=(-1.0, 1.0))
    creator.create(
        'Individual', gp.PrimitiveTree, fitness=creator.FitnessMulti
    )

    toolbox = base.Toolbox()
    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register(
        'individual', tools.initIterate, creator.Individual, toolbox.expr
    )
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', gp.compile, pset=pset)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)

    stats = tools.Statistics(get_fitness_attr)
    stats.register('Minimum score', np.min)
    stats.register('Average score', np.mean)
    stats.register('Maximum score', np.max)

    toolbox.register('select', _combined_selection_operator)

    halloffame = tools.ParetoFront(similar=pareto_eq)
    _model = partial(
        algorithms.eaSimple,
        toolbox=toolbox,
        stats=stats,
        halloffame=halloffame,
        verbose=True,
    )

    def _random_mutation_operator(self, individual):
        roll = random()
        if roll <= 0.333333:
            return gp.mutUniform(
                individual, expr=self.toolbox.expr_mut, pset=self.pset
            )
        elif roll <= 0.666666:
            return gp.mutInsert(individual, pset=self.pset)
        else:
            return gp.mutShrink(individual)


def prepare_dataframe(data_source, class_column=default_class_column):
    data_source = DataFrame(data_source)
    cats = data_source[class_column].unique().tolist()
    data_source[class_column] = data_source[class_column].apply(cats.index)

    _, testing_indices = train_test_split(
        data_source.index,
        stratify=data_source[class_column].values,
        train_size=.75,
        test_size=.25,
    )

    data_source = data_source.set_index(class_column)
    data_source.index = MultiIndex.from_tuples(
       [(i not in testing_indices, c) for i, c in enumerate(data_source.index)]
    )

    return data_source


def get_fitness_attr(x): return x.fitness.values[1]


from ipywidgets import HTML
class TPOT(ClassifierMixin, BaseEstimator, DeapSetup):
    viewer = HTML()
    def __init__(
        self, models=[], population=10, generations=10,
        mutation_rate=0.9, crossover_rate=0.05, random_state=0,
        class_column=default_class_column,
    ):
        self.models = models
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.class_column = class_column

        self.toolbox.register('mutate', self._random_mutation_operator)

    def fit(self, X, **kwargs):
        self.set_params(**kwargs)
        for model in [*self.base_models, *self.models]:
            if model.__name__ not in self.pset.mapping:
                self.pset.addPrimitive(
                    model.fit_predict,
                    [*self.input_types, *model.trait_types()],
                    model.output_type,
                    name=model.__name__,
                )

        self.data_source = prepare_dataframe(X, self.class_column)

        pop = self.toolbox.population(
            n=self.population
        )
        self.toolbox.register(
            'evaluate', self.evaluate,
            data_source=self.data_source
        )

        pop = self._model(
            population=pop,
            ngen=self.generations,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
        )

        top_score = 0.0
        for pipeline in self.halloffame:
            pipeline_score = PipelineEstimator(
                self.toolbox.compile(expr=pipeline)
            ).score(self.data_source.ix[False])
            if pipeline_score > top_score:
                top_score = pipeline_score
                self._optimized_pipeline = pipeline

    def predict(self, X):
        return self.toolbox.compile(
            self._optimized_pipeline
        )(X)

    def evaluate(self, individual, data_source):
        self.viewer.value = """<pre><code>{}</code></pre>""".format(
            individual.__str__(),
        )
        model = PipelineEstimator(
            self.toolbox.compile(expr=individual)
        )
        # inherited from ClassifierMixin
        score = 1.

        try:
            # print(individual)
            if hasattr(model, 'score'):
                score = model.score(
                    data_source,
                )
        except MemoryError:
            # Throw out GP expressions that are too large to be
            # compiled in Python
            return 5000., 0.
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            # Catch-all: Do not allow one pipeline that crashes to cause
            # TPOT to crash Instead assign the crashing pipeline a poor fitness
            return 5000., 0.

        operator_count = 0
        for i in range(len(individual)):
            node = individual[i]
            if type(node) is gp.Terminal:
                continue
            if type(node) is gp.Primitive and node.name in [
                'add', 'sub', 'mul', '_div', '_combine_dfs'
            ]:
                continue

            operator_count += 1
        if isinstance(score, (float, np.float64, np.float32)):
            return max(1, operator_count), score
        else:
            raise ValueError('Scoring function does not return a float')

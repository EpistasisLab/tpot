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
from models.base import (
    PipelineEstimator,
)
from models.decomposition import (
    fast_ica,
)
from models.ensemble import (
    random_forest,
    ada_boost,
)
from toolz import (
    compose,
    partial,
    second,
)

import operator


class DeapSetup(object):
    base_models = [
        random_forest, ada_boost, fast_ica,
    ]

    @staticmethod
    def get_fitness_attr(x): return x.fitness.values[1]

    @staticmethod
    def _div(num1, num2):
        return float(num1) / float(num2) if num2 != 0. else 0.

    @staticmethod
    def _combined_selection_operator(self, individuals, k):
        return tools.selNSGA2(individuals, int(k / 5.)) * 5

    @staticmethod
    def _combine_dfs(input_df1, input_df2):
        for column in input_df2.columns:
            input_df1[column] = input_df2[column]
        return input_df1

    @staticmethod
    def _zero_count(input_df):
        modified_df = input_df.copy()
        modified_df['non_zero'] = modified_df.apply(
            np.count_nonzero, axis=1
        ).astype(np.float64)
        modified_df['zero_col'] = len(modified_df)-modified_df['non_zero']
        return modified_df.copy()

    # This can be changed for numpy arrays later
    input_types = [DataFrame]

    # Similarity function for the Pareto Front Hall of Frame
    @staticmethod
    def pareto_eq(a, b):
        return np.all(get_fitness_attr(a) == get_fitness_attr(b))

    pset = gp.PrimitiveSetTyped('MAIN', [DataFrame], DataFrame)
    pset.addPrimitive(operator.add, [int, int], int)
    pset.addPrimitive(operator.sub, [int, int], int)
    pset.addPrimitive(operator.mul, [int, int], int)

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

    pset.addPrimitive(
        _combine_dfs, [DataFrame, DataFrame], DataFrame
    )
    pset.addPrimitive(_zero_count, [DataFrame], DataFrame)
    pset.addPrimitive(_div, [int, int], float)

    toolbox.register('select', _combined_selection_operator)

    halloffame = tools.ParetoFront(similar=pareto_eq)
    _model = partial(
        algorithms.eaSimple,
        toolbox=toolbox,
        stats=stats,
        halloffame=halloffame,
    )

    def __init__(self, data_source, class_column='class',
                 mutation_rate=0.9, crossover_rate=0.05,
                 random_state=0, verbose=2):

        self.data_source = prepare_dataframe(data_source)

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


def prepare_dataframe(data_source, class_column='species'):
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


class teapot(ClassifierMixin, BaseEstimator, DeapSetup):
    def __init__(
        self, models=[], population=10, generations=10,
        mutation_rate=0.9, crossover_rate=0.05, random_state=0,
    ):
        self.models = self.models
        self.population = population
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state

        self.toolbox.register('mutate', self._random_mutation_operator)

    def fit(self, X):
        for model in [self.base_models, self.models]:
            self.pset.addPrimitive(
                model.evaluate,
                [*self.input_types, *model.trait_types()],
                model.output_type,
                name=model.__name__,
            )

        self.data_source = prepare_dataframe(X)

        pop = self.toolbox.population(
            n=self.population
        )
        self.toolbox.register('evaluate', self.evaluate)

        pop = self._model(
            population=pop,
            ngen=self.generations,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
        )

        top_score = 0.0
        scoring = compose(second, self.evaluate)
        for pipeline in self.halloffame:
            pipeline_score = scoring(pipeline)
            if pipeline_score > top_score:
                top_score = pipeline_score
                self._optimized_pipeline = pipeline
                self._best_model = self.toolbox.compile(
                    self._optimized_pipeline
                )

    def evaluate(self, individual, *args):
        try:
            model = PipelineEstimator.compose(
                self.toolbox.compile(expr=individual), *args
            )
            score = model.score(self.data_source.ix[False])
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

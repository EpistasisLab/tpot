import numpy as np
from random import choice
from toolz.curried import *
from deap import (
    creator,
)
from deap.algorithms import (
    eaSimple,
)
from deap.base import (
    Fitness, Toolbox,
)
from deap.gp import (
    cxOnePoint, genFull, genHalfAndHalf, graph,
    mutInsert, mutShrink, mutUniform,
    Primitive, PrimitiveSetTyped, PrimitiveTree,
)
from deap.tools import (
    initRepeat, initIterate, selNSGA2,
    History, ParetoFront, Statistics,
)
from sklearn.base import (
    BaseEstimator, ClassifierMixin, TransformerMixin,
)
from sklearn.pipeline import (
    make_pipeline, make_union,
)


class Tree(PrimitiveTree):
    @property
    def model(self):
        return self.compile_(0)

    def compile_(self, model_id=0):
        """Compile an individual graph of sklearn models in a single model."""

        node, edge, label = graph(self)

        # Express the labels from the graph as their function values.
        label = valmap(
                lambda x: self.pset.context.get(x)
                if x in self.pset.context else x, label
            )

        # In reverse order, insert each model into a sklearn pipeline.  No
        # functions are build.  Only a pipeline is created.
        for junction in pipe(
            self,
            map(partial(flip(isinstance), Primitive)),
            list, np.where, first, reversed,
        ):
            # first element is a function
            # second element is a param.
            label[junction] = label[junction](
                *pipe(
                    edge,
                    filter(compose(lambda v: junction == v, first)),
                    map(second), map(label.get), list
                )
            )

        # `0` is the model_id for the main model.
        return label.get(model_id)


class tpot_experimental(BaseEstimator):
    def __init__(
        self, crossover_rate=.1, expr=genHalfAndHalf, generation=10,
        max_=3, min_=0, models=[], mutation_rate=.9, population=50,
        select=selNSGA2,
    ):
        self.crossover_rate = crossover_rate
        self.errors_ = [0]*2
        self.expr = expr
        self.generation = generation
        self.max_ = max_
        self.min_ = min_
        self.models = models
        self.mutation_rate = mutation_rate
        self.population = population
        self.select = select

    def evaluate(self, individual, df):
        """"""
        model = individual.model

        # Fit the model and catch and fitting errors
        try:
            model.fit(df.ix[True].values, df.ix[True].index)
        except:
            self.errors_[0] += 1
            return (len(individual), 0.)

        # Predict the model and catch and scoring errors.
        try:
            score = model.score(
                df.values, df.index.get_level_values(1)
            )
            return (len(individual), score)
        except:
            # score error
            self.errors_[1] += 1
            return (len(individual), 0.)

    def fit(self, df, **kwargs):
        self.set_params(**kwargs)
        pset = self.pset = primitives(self.models)

        creator.create('FitnessMulti', Fitness, weights=(-1.0, 1.0))

        creator.create(
            'Individual', Tree, pset=pset, fitness=creator.FitnessMulti,
        )

        toolbox = self.toolbox_ = Toolbox()
        toolbox.register(
            'expr', self.expr, pset=pset, min_=self.min_, max_=self.max_
        )
        toolbox.register(
            'individual', initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register('population', initRepeat, list, toolbox.individual)

        toolbox.register(
            'evaluate', self.evaluate, df=df,
        )
        toolbox.register('expr_mut', genFull, min_=self.min_, max_=self.max_)
        toolbox.register('mate', cxOnePoint)
        toolbox.register(
            'mutate',
            lambda individual: pipe(
                individual, [
                    partial(mutUniform, expr=toolbox.expr_mut, pset=pset),
                    partial(mutInsert, pset=pset),
                    partial(mutShrink)
                ][choice(range(3))],
            )
        )
        toolbox.register('select', self.select)

        halloffame = self.halloffame_ = ParetoFront()

        stats = self.stats_ = Statistics(
            compose(second, lambda x: x.fitness.values)
        )
        stats.register('Minimum score', np.min)
        stats.register('Average score', np.mean)
        stats.register('Maximum score', np.max)

        pop = toolbox.population(
            n=self.population
        )

        history = self.history_ = History()
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)

        history.update(pop)

        self.result_ = eaSimple(
            cxpb=self.crossover_rate,
            halloffame=halloffame,
            mutpb=self.mutation_rate,
            ngen=self.generation,
            population=pop,
            stats=stats,
            toolbox=toolbox,
            verbose=True,
        )


def primitives(models):
    pset = PrimitiveSetTyped('main', [], ClassifierMixin)
    for model in models:
        if isinstance(
            model, (ClassifierMixin, TransformerMixin)
        ):
            pset.addTerminal(
                model,
                ClassifierMixin if isinstance(
                    model, ClassifierMixin
                ) else TransformerMixin,
            )
    for junction in [make_pipeline, make_union]:
        pset.addPrimitive(
            junction, [TransformerMixin]*2, TransformerMixin
        )
    pset.addPrimitive(
        make_pipeline,
        [TransformerMixin, ClassifierMixin],
        ClassifierMixin, name='classify'
    )
    return pset

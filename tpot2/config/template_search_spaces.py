import tpot2
from tpot2.search_spaces.pipelines import *
from tpot2.search_spaces.nodes import *
from .get_configspace import get_search_space



def get_linear_search_space(classification=True, inner_predictors=True, **get_search_space_params ):

    if classification:
        selectors = get_search_space(["selectors","selectors_classification", "Passthrough"], **get_search_space_params)
        estimators = get_search_space(["classifiers"], **get_search_space_params)
    else:
        selectors = get_search_space(["selectors","selectors_regression", "Passthrough"], **get_search_space_params)
        estimators = get_search_space(["regressors"], **get_search_space_params)

    # this allows us to wrap the classifiers in the EstimatorTransformer
    # this is necessary so that classifiers can be used inside of sklearn pipelines
    wrapped_estimators = WrapperPipeline(tpot2.builtin_modules.EstimatorTransformer, {}, estimators)

    scalers = get_search_space(["scalers","Passthrough"], **get_search_space_params)

    transformers_layer =UnionPipeline([
                            ChoicePipeline([
                                DynamicUnionPipeline(get_search_space(["transformers"],**get_search_space_params)),
                                get_search_space("SkipTransformer", **get_search_space_params),
                            ]),
                            get_search_space("Passthrough", **get_search_space_params)
                            ]
                        )
    
    inner_estimators_layer = UnionPipeline([
                                ChoicePipeline([
                                    DynamicUnionPipeline(wrapped_estimators),
                                    get_search_space("SkipTransformer",  **get_search_space_params),
                                ]),
                                get_search_space("Passthrough",  **get_search_space_params)]
                            )

    if inner_predictors:
        search_space = SequentialPipeline(search_spaces=[
                                            scalers,
                                            selectors, 
                                            transformers_layer,
                                            inner_estimators_layer,
                                            estimators,
                                            ])
    else:
        search_space = SequentialPipeline(search_spaces=[
                                            scalers,
                                            selectors, 
                                            transformers_layer,
                                            estimators,
                                            ])

    return search_space


def get_graph_search_space(classification=True, inner_predictors=True, **get_search_space_params ):

    if classification:
        root_search_space = get_search_space(["classifiers"], **get_search_space_params)
        inner_search_space = tpot2.config.get_search_space(["transformers","scalers","selectors_classification"],**get_search_space_params)
    else:
        root_search_space = get_search_space(["regressors"], **get_search_space_params)
        

    if classification:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(["classifiers","transformers","scalers","selectors_classification"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["transformers","scalers","selectors_classification"],**get_search_space_params)
    else:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(["regressors", "transformers","scalers","selectors_regression"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["transformers","scalers","selectors_regression"],**get_search_space_params)


    search_space = tpot2.search_spaces.pipelines.GraphPipeline(
        root_search_space= root_search_space,
        leaf_search_space = None, 
        inner_search_space = inner_search_space,
    )

    return search_space


def get_graph_search_space_light(classification=True, inner_predictors=True, **get_search_space_params ):

    if classification:
        root_search_space = get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB'], **get_search_space_params)
    else:
        root_search_space = get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV"], **get_search_space_params)
        

    if classification:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB',"transformers","scalers","SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["transformers","scalers","SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
    else:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV", "transformers","scalers", "SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["transformers", "scalers", "SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)


    search_space = tpot2.search_spaces.pipelines.GraphPipeline(
        root_search_space= root_search_space,
        leaf_search_space = None, 
        inner_search_space = inner_search_space,
    )

    return search_space


def get_light_search_space(classification=True, inner_predictors=False, **get_search_space_params ):

    selectors = get_search_space(["SelectFwe", "SelectPercentile", "VarianceThreshold","Passthrough"], **get_search_space_params)

    if classification:
        estimators = get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB'], **get_search_space_params)
    else:
        estimators = get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV"], **get_search_space_params)

    # this allows us to wrap the classifiers in the EstimatorTransformer
    # this is necessary so that classifiers can be used inside of sklearn pipelines
    wrapped_estimators = WrapperPipeline(tpot2.builtin_modules.EstimatorTransformer, {}, estimators)

    scalers = get_search_space(["scalers","Passthrough"], **get_search_space_params)

    transformers_layer =UnionPipeline([
                            ChoicePipeline([
                                DynamicUnionPipeline(get_search_space(["transformers"],**get_search_space_params)),
                                get_search_space("SkipTransformer", **get_search_space_params),
                            ]),
                            get_search_space("Passthrough", **get_search_space_params)
                            ]
                        )
    
    inner_estimators_layer = UnionPipeline([
                                ChoicePipeline([
                                    DynamicUnionPipeline(wrapped_estimators),
                                    get_search_space("SkipTransformer",  **get_search_space_params),
                                ]),
                                get_search_space("Passthrough",  **get_search_space_params)]
                            )

    if inner_predictors:
        search_space = SequentialPipeline(search_spaces=[
                                            scalers,
                                            selectors, 
                                            transformers_layer,
                                            inner_estimators_layer,
                                            estimators,
                                            ])
    else:
        search_space = SequentialPipeline(search_spaces=[
                                            scalers,
                                            selectors, 
                                            transformers_layer,
                                            estimators,
                                            ])

    return search_space

def get_mdr_search_space(classification=True, **get_search_space_params ):

    mdr_sp = DynamicLinearPipeline(get_search_space(["ReliefF", "SURF", "SURFstar", "MultiSURF", "ContinuousMDR"], **get_search_space_params), max_length=10)

    if classification:
        estimators = get_search_space(['LogisticRegression'], **get_search_space_params)
    else:
        estimators = get_search_space(["ElasticNetCV"], **get_search_space_params)

    search_space = SequentialPipeline(search_spaces=[
                                            mdr_sp,
                                            estimators,
                                            ])

    return search_space




def get_template_search_spaces(default_search_space, classification=True, inner_predictors=None, **get_search_space_params):
    
    if inner_predictors is None:
        if default_search_space == "light" or default_search_space == "graph_light":
            inner_predictors = False
        else:
            inner_predictors = True
    
    if isinstance(default_search_space, str):
        if default_search_space == "linear":
            return get_linear_search_space(classification, inner_predictors, **get_search_space_params)
        elif default_search_space == "graph":
            return get_graph_search_space(classification, inner_predictors, **get_search_space_params)
        elif default_search_space == "graph-light":
            return get_graph_search_space_light(classification, inner_predictors, **get_search_space_params)
        elif default_search_space == "linear-light":
            return get_light_search_space(classification, inner_predictors, **get_search_space_params)
        elif default_search_space == "mdr":
            return get_mdr_search_space(classification, **get_search_space_params)
        else:
            raise ValueError("Invalid search space")
    else:
        return default_search_space
import tpot2
from tpot2.search_spaces.pipelines import *
from tpot2.search_spaces.nodes import *
from tpot2.config import get_search_space



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
        inner_search_space = tpot2.config.get_search_space(["selectors","transformers","scalers","selectors_classification"],**get_search_space_params)
    else:
        root_search_space = get_search_space(["regressors"], **get_search_space_params)
        

    if classification:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(["classifiers", "selectors","transformers","scalers","selectors_regression"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["selectors","transformers","scalers","selectors_regression"],**get_search_space_params)
    else:
        if inner_predictors:
            inner_search_space = tpot2.config.get_search_space(["regressors", "selectors","transformers","scalers","selectors_regression"],**get_search_space_params)
        else:
            inner_search_space = tpot2.config.get_search_space(["selectors","transformers","scalers","selectors_regression"],**get_search_space_params)


    search_space = tpot2.search_spaces.pipelines.GraphPipeline(
        root_search_space= root_search_space,
        leaf_search_space = None, 
        inner_search_space = inner_search_space,
    )

    return search_space

def get_default_search_space(default_search_space, classification=True, inner_predictors=True, **get_search_space_params):
    if isinstance(default_search_space, str):
        if default_search_space == "linear":
            return get_linear_search_space(classification, inner_predictors, **get_search_space_params)
        elif default_search_space == "graph":
            return get_graph_search_space(classification, inner_predictors, **get_search_space_params)
        else:
            raise ValueError("Invalid search space")
    else:
        return default_search_space
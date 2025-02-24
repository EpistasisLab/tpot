"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

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
import tpot
from tpot.search_spaces.pipelines import *
from tpot.search_spaces.nodes import *
from .get_configspace import get_search_space
import sklearn.model_selection
import sklearn


def get_linear_search_space(classification=True, inner_predictors=True, cross_val_predict_cv=0, **get_search_space_params ):

    if classification:
        selectors = get_search_space(["selectors","selectors_classification", "Passthrough"], **get_search_space_params)
        estimators = get_search_space(["classifiers"], **get_search_space_params)
    else:
        selectors = get_search_space(["selectors","selectors_regression", "Passthrough"], **get_search_space_params)
        estimators = get_search_space(["regressors"], **get_search_space_params)

    # this allows us to wrap the classifiers in the EstimatorTransformer
    # this is necessary so that classifiers can be used inside of sklearn pipelines
    wrapped_estimators = WrapperPipeline(tpot.builtin_modules.EstimatorTransformer, {'cross_val_predict_cv':cross_val_predict_cv}, estimators)

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


def get_graph_search_space(classification=True, inner_predictors=True, cross_val_predict_cv=0, **get_search_space_params ):

    if classification:
        root_search_space = get_search_space(["classifiers"], **get_search_space_params)
        inner_search_space = tpot.config.get_search_space(["transformers","scalers","selectors_classification"],**get_search_space_params)
    else:
        root_search_space = get_search_space(["regressors"], **get_search_space_params)
        

    if classification:
        if inner_predictors:
            inner_search_space = tpot.config.get_search_space(["classifiers","transformers","scalers","selectors_classification"],**get_search_space_params)
        else:
            inner_search_space = tpot.config.get_search_space(["transformers","scalers","selectors_classification"],**get_search_space_params)
    else:
        if inner_predictors:
            inner_search_space = tpot.config.get_search_space(["regressors", "transformers","scalers","selectors_regression"],**get_search_space_params)
        else:
            inner_search_space = tpot.config.get_search_space(["transformers","scalers","selectors_regression"],**get_search_space_params)


    search_space = tpot.search_spaces.pipelines.GraphSearchPipeline(
        root_search_space= root_search_space,
        leaf_search_space = None, 
        inner_search_space = inner_search_space,
        cross_val_predict_cv=cross_val_predict_cv,
        max_size=15,
    )

    return search_space


def get_graph_search_space_light(classification=True, inner_predictors=True, cross_val_predict_cv=0, **get_search_space_params ):

    if classification:
        root_search_space = get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB'], **get_search_space_params)
    else:
        root_search_space = get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV"], **get_search_space_params)
        

    if classification:
        if inner_predictors:
            inner_search_space = tpot.config.get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB',"transformers","scalers","SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
        else:
            inner_search_space = tpot.config.get_search_space(["transformers","scalers","SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
    else:
        if inner_predictors:
            inner_search_space = tpot.config.get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV", "transformers","scalers", "SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)
        else:
            inner_search_space = tpot.config.get_search_space(["transformers", "scalers", "SelectFwe", "SelectPercentile", "VarianceThreshold"],**get_search_space_params)


    search_space = tpot.search_spaces.pipelines.GraphSearchPipeline(
        root_search_space= root_search_space,
        leaf_search_space = None, 
        inner_search_space = inner_search_space,
        cross_val_predict_cv=cross_val_predict_cv,
        max_size=15,
    )

    return search_space


def get_light_search_space(classification=True, inner_predictors=False, cross_val_predict_cv=0, **get_search_space_params ):

    selectors = get_search_space(["SelectFwe", "SelectPercentile", "VarianceThreshold","Passthrough"], **get_search_space_params)

    if classification:
        estimators = get_search_space(['BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'KNeighborsClassifier', 'LogisticRegression', 'MultinomialNB'], **get_search_space_params)
    else:
        estimators = get_search_space(["RidgeCV", "LinearSVR", "LassoLarsCV", "KNeighborsRegressor", "DecisionTreeRegressor", "ElasticNetCV"], **get_search_space_params)

    # this allows us to wrap the classifiers in the EstimatorTransformer
    # this is necessary so that classifiers can be used inside of sklearn pipelines
    wrapped_estimators = WrapperPipeline(tpot.builtin_modules.EstimatorTransformer, {'cross_val_predict_cv':cross_val_predict_cv}, estimators)

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

    

    if classification:
        mdr_sp = DynamicLinearPipeline(get_search_space(["ReliefF", "SURF", "SURFstar", "MultiSURF", "MDR"], **get_search_space_params), max_length=10)
        estimators = get_search_space(['LogisticRegression'], **get_search_space_params)
    else:
        mdr_sp = DynamicLinearPipeline(get_search_space(["ReliefF", "SURF", "SURFstar", "MultiSURF", "ContinuousMDR"], **get_search_space_params), max_length=10)
        estimators = get_search_space(["ElasticNetCV"], **get_search_space_params)

    search_space = SequentialPipeline(search_spaces=[
                                            mdr_sp,
                                            estimators,
                                            ])

    return search_space




def get_template_search_spaces(search_space, classification=True, inner_predictors=None, cross_val_predict_cv=None, **get_search_space_params):
    """
    Returns a search space which can be optimized by TPOT.

    Parameters
    ----------
    search_space: str or SearchSpace
        The default search space to use. If a string, it should be one of the following:
            - 'linear': A search space for linear pipelines
            - 'linear-light': A search space for linear pipelines with a smaller, faster search space
            - 'graph': A search space for graph pipelines
            - 'graph-light': A search space for graph pipelines with a smaller, faster search space
            - 'mdr': A search space for MDR pipelines
        If a SearchSpace object, it should be a valid search space object for TPOT.
    
    classification: bool, default=True
        Whether the problem is a classification problem or a regression problem.

    inner_predictors: bool, default=None
        Whether to include additional classifiers/regressors before the final classifier/regressor (allowing for ensembles). 
        Defaults to False for 'linear-light' and 'graph-light' search spaces, and True otherwise. (Not used for 'mdr' search space)
    
    cross_val_predict_cv: int, default=None
        The number of folds to use for cross_val_predict. 
        Defaults to 0 for 'linear-light' and 'graph-light' search spaces, and 5 otherwise. (Not used for 'mdr' search space)

    get_search_space_params: dict
        Additional parameters to pass to the get_search_space function.
    
    """
    if inner_predictors is None:
        if search_space == "light" or search_space == "graph_light":
            inner_predictors = False
        else:
            inner_predictors = True

    if cross_val_predict_cv is None:
        if search_space == "light" or search_space == "graph_light":
            cross_val_predict_cv = 0
        else:
            if classification:
                cross_val_predict_cv = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            else:
                cross_val_predict_cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    
    if isinstance(search_space, str):
        if search_space == "linear":
            return get_linear_search_space(classification, inner_predictors, cross_val_predict_cv=cross_val_predict_cv, **get_search_space_params)
        elif search_space == "graph":
            return get_graph_search_space(classification, inner_predictors, cross_val_predict_cv=cross_val_predict_cv, **get_search_space_params)
        elif search_space == "graph-light":
            return get_graph_search_space_light(classification, inner_predictors, cross_val_predict_cv=cross_val_predict_cv, **get_search_space_params)
        elif search_space == "linear-light":
            return get_light_search_space(classification, inner_predictors, cross_val_predict_cv=cross_val_predict_cv, **get_search_space_params)
        elif search_space == "mdr":
            return get_mdr_search_space(classification, **get_search_space_params)
        else:
            raise ValueError("Invalid search space")
    else:
        return search_space
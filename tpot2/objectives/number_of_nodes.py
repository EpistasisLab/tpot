from ..graphsklearn import GraphPipeline
from sklearn.pipeline import Pipeline
import sklearn

def number_of_nodes_objective(est):
    if isinstance(est, GraphPipeline):
        return sum(number_of_nodes_objective(est.graph.nodes[node]["instance"]) for node in est.graph.nodes)
    if isinstance(est, Pipeline):
        return sum(number_of_nodes_objective(estimator) for _,estimator in est.steps)
    if isinstance(est, sklearn.pipeline.FeatureUnion):
        return sum(number_of_nodes_objective(estimator) for _,estimator in est.transformer_list)
    
    return 1
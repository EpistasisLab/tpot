from tpot2.representations.graph_pipeline_individual.individual import *
import optuna
import numpy as np
import copy
import dask
import traceback


# labels all nodes in the graph with a unique ID.
# This allows use to identify exact nodes in the copies on the graph.
# This is necessary since copies of the graph use different NodeLabel object instances as keys, making it hard to identify which are the same nodes.
def label_nodes_in_graphindividual(graphindividual):
    nodes_list = graphindividual.full_node_list()
    for i, node in enumerate(nodes_list):
        if not isinstance(node, NodeLabel):
            continue
        else:
            node.label = f'node_{i}'


def optuna_optimize_full_graph(graphindividual, objective_function, objective_function_weights, steps=5, relabel=True, verbose=0, max_eval_time_seconds=60*5, max_time_seconds=60*20, n_returned_models='all', study=None, **objective_kwargs):
    if relabel:
        label_nodes_in_graphindividual(graphindividual)
    
    graphindividual = copy.deepcopy(graphindividual)
    nodes_list = graphindividual.full_node_list()


    nodes_to_optimize = []
    for node in nodes_list:
        if not isinstance(node, NodeLabel) or isinstance(graphindividual.select_config_dict(node)[node.method_class],dict):
            continue
        else:
            nodes_to_optimize.append(node)

    def objective(trial):
        param_dict = dict()
        graphindividual.key = None
        for node in nodes_to_optimize:
            params = graphindividual.select_config_dict(node)[node.method_class](trial, name=node.label)
            node.hyperparameters = params
            param_dict[node.label] = params
        
        trial.set_user_attr('params', param_dict)
        
        try:
            scores = tpot2.objective_nan_wrapper(graphindividual, objective_function, verbose=verbose,timeout=max_eval_time_seconds,**objective_kwargs)#objective_function(graphindividual)
            trial.set_user_attr('scores', list(scores))
            if scores[0] != "INVALID" and scores[0] != "TIMEOUT":
                scores = np.array(scores) * objective_function_weights
            scores = list(scores)
            
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            scores = ['INVALID']
            trial.set_user_attr('scores', scores)
        return scores

    study.optimize(objective, n_trials=steps, timeout=max_time_seconds)
   
    return study

def get_all_individuals_from_study(study, graphindividual, n_returned_models='all'):
    all_graphs = []
    all_scores = []

    if n_returned_models == 'pareto':
        trials_list = study.best_trials
    else:
        trials_list = study.trials

    for trial in trials_list:
        if not 'scores' in trial.user_attrs:
            continue

        params = trial.user_attrs['params']
        scores = trial.user_attrs['scores']

        newgraphindividual = copy.deepcopy(graphindividual)
        newgraphindividual.key = None
        for node in newgraphindividual.full_node_list():
            if not isinstance(node, tpot2.NodeLabel):
                continue
            else:
                node.hyperparameters = params[node.label]

        all_graphs.append(newgraphindividual)
        all_scores.append(scores)
   
    return all_graphs, all_scores


def simple_parallel_optuna(individuals,  objective_function, objective_function_weights, client, storage, steps=5, verbose=0, max_eval_time_seconds=60*5, max_time_seconds=60*20, **objective_kwargs):
    num_workers = len(client.scheduler_info()['workers'])
    worker_per_individual = max(1,int(np.floor(num_workers/len(individuals))))
    remainder = num_workers%len(individuals)

    print(worker_per_individual)
    print(remainder)

    directions = np.repeat('maximize',len(objective_function_weights))

    backend_storage = optuna.storages.InMemoryStorage()
    dask_storage = optuna.integration.DaskStorage(storage=backend_storage, client=client)

    futures = []
    studies = []
    for i, ind in enumerate(individuals):
        label_nodes_in_graphindividual(ind)
        #study  = optuna.create_study(directions=directions, storage=f"{storage}", load_if_exists=False)
        study  = optuna.create_study(directions=directions, storage=dask_storage, load_if_exists=False)
        studies.append(study)
        if i == 0:
            n_futures = worker_per_individual + remainder
        else:
            n_futures = worker_per_individual
        
        trials_per_thread = int(np.ceil(steps/n_futures))
        for _ in range(n_futures):
            futures.append(client.submit(optuna_optimize_full_graph, graphindividual=ind, objective_function=objective_function, objective_function_weights=objective_function_weights, steps=trials_per_thread, verbose=verbose, max_eval_time_seconds=max_eval_time_seconds, max_time_seconds=max_time_seconds, study=study, relabel=False, pure=False, **objective_kwargs))

    
    dask.distributed.wait(futures)

    all_graphs = []
    all_scores = []
    for study, ind in zip(studies,individuals):
        graphs, scores = get_all_individuals_from_study(study, ind)
        all_graphs.extend(graphs)
        all_scores.extend(scores)

    return all_graphs, all_scores
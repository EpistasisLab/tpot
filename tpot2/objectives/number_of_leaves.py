def number_of_leaves_scorer(est,X=None, y=None):
    return len([v for v, d in est.graph.out_degree() if d == 0])

def number_of_leaves_objective(est):
    """
    Calculates the number of leaves (input nodes) in a GraphPipeline

    Parameters
    ----------
    est: GraphPipeline
        The pipeline to compute the number of leaves for
    """
    return len([v for v, d in est.graph.out_degree() if d == 0])
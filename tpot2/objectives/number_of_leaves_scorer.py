def number_of_leaves_scorer(est,X,y):
    return len([v for v, d in est.graph.out_degree() if d == 0])
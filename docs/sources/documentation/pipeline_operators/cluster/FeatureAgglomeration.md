# Agglomerate Features
* * *

Uses Scikit-learn's FeatureAgglomeration to transform the feature set.

## Dependencies
    sklearn.cluster.FeatureAgglomeration


Parameters
----------
    input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
        Input DataFrame to scale
    n_clusters: int
        The number of clusters to find.
    affinity: int
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted.
        Input integer is used to select one of the above strings.
    linkage: int
        Can be one of the following values:
            "ward", "complete", "average"
        Input integer is used to select one of the above strings.

Returns
-------
    modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
        Returns a DataFrame containing the transformed features

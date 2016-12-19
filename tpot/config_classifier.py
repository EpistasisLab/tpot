# TODO: figure out xg_boost because it does not import directly from sklearn clf class
classifier_config_dict = {

    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    },

    'sklearn.naive_bayes.GaussianNB': {
        "criterion": ['gini', 'entropy'],
        "max_features": [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "bootstrap": [True, False]
    },

    'sklearn.ensemble.ExtraTreesClassifier': {
        "criterion": ["gini", "entropy"],
        "max_features": [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "bootstrap": [True, False]
    },

    'sklearn.ensemble.GradientBoostingClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'subsample': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                     0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                     0.9,  0.95,  1.],
        'max_features': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.]
    },

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },


    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'criterion': ["gini", "entropy"],
        'max_features': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        'bootstrap': [True, False]
    },

    # Preprocessors
    'sklearn.preprocessing.Binarizer': {
        'threshold': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                      0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                      0.9,  0.95,  1.]
    },

    'sklearn.decomposition.FastICA': {
        'tol': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                0.9,  0.95,  1.]
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    },

    'sklearn.preprocessing.MaxAbsScaler': {

    },

    'sklearn.preprocessing.MinMaxScaler': {

    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                  0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                  0.9,  0.95,  1.],
        'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': 'randomized',
        'iterated_power': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': 2,
        'include_bias': False,
        'interaction_only': False
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                  0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                  0.9,  0.95,  1.]
    },

    'sklearn.preprocessing.RobustScaler': {

    },

    'sklearn.preprocessing.StandardScaler': {

    },

    'tpot.operators.preprocessors.ZeroCount': {

    },

    # Selectors

}



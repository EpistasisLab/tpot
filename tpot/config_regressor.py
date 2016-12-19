# TODO: figure out xg_boost because it does not import directly from sklearn clf class

regressor_config_dict = {

    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                    0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                     0.9,  0.95,  1.],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        'max_features': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'subsample':[0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                    0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                    0.9,  0.95,  1.],
        'max_features': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    },

    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'max_features': [0.,  0.05,  0.1,  0.15,  0.2,  0.25,  0.3,  0.35,  0.4,
                        0.45,  0.5,  0.55,  0.6,  0.65,  0.7,  0.75,  0.8,  0.85,
                        0.9,  0.95,  1.],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {}
}
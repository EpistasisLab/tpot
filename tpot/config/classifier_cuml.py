# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

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

import numpy as np

# This configuration provides users with access to a GPU the ability to
# use RAPIDS cuML and DMLC/XGBoost classifiers as estimators alongside
# the scikit-learn preprocessors in the TPOT default configuration.

classifier_config_cuml = {
    # cuML + DMLC/XGBoost Classifiers

    "cuml.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform",],
    },

    "cuml.linear_model.LogisticRegression": {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.,],
    },

    "xgboost.XGBClassifier": {
        "n_estimators": [100],
        "max_depth": range(3, 10),
        "learning_rate": [1e-2, 1e-1, 0.5, 1.],
        "subsample": np.arange(0.05, 1.01, 0.05),
        "min_child_weight": range(1, 21),
        "alpha": [1, 10],
        "tree_method": ["gpu_hist"],
        "n_jobs": [1],
        "verbosity": [0]
    },

    # Sklearn Preprocesssors

    "sklearn.preprocessing.Binarizer": {
        "threshold": np.arange(0.0, 1.01, 0.05)
    },

    "sklearn.decomposition.FastICA": {
        "tol": np.arange(0.0, 1.01, 0.05)
    },

    "sklearn.cluster.FeatureAgglomeration": {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"]
    },

    "sklearn.preprocessing.MaxAbsScaler": {
    },

    "sklearn.preprocessing.MinMaxScaler": {
    },

    "sklearn.preprocessing.Normalizer": {
        "norm": ["l1", "l2", "max"]
    },

    "sklearn.kernel_approximation.Nystroem": {
        "kernel": ["rbf", "cosine", "chi2", "laplacian", "polynomial", "poly", "linear", "additive_chi2", "sigmoid"],
        "gamma": np.arange(0.0, 1.01, 0.05),
        "n_components": range(1, 11)
    },

    "sklearn.decomposition.PCA": {
        "svd_solver": ["randomized"],
        "iterated_power": range(1, 11)
    },

    "sklearn.kernel_approximation.RBFSampler": {
        "gamma": np.arange(0.0, 1.01, 0.05)
    },

    "sklearn.preprocessing.RobustScaler": {
    },

    "sklearn.preprocessing.StandardScaler": {
    },

    "tpot.builtins.ZeroCount": {
    },

    "tpot.builtins.OneHotEncoder": {
        "minimum_fraction": [0.05, 0.1, 0.15, 0.2, 0.25],
        "sparse": [False],
        "threshold": [10]
    },

    # Selectors

    "sklearn.feature_selection.SelectFwe": {
        "alpha": np.arange(0, 0.05, 0.001),
        "score_func": {
            "sklearn.feature_selection.f_classif": None
        }
    },

    "sklearn.feature_selection.SelectPercentile": {
        "percentile": range(1, 100),
        "score_func": {
            "sklearn.feature_selection.f_classif": None
        }
    },

    "sklearn.feature_selection.VarianceThreshold": {
        "threshold": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    }
}

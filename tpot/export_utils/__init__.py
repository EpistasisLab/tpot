# -*- coding: utf-8 -*-

'''
Copyright 2016 Randal S. Olson

This file is part of the TPOT library.

The TPOT library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The TPOT library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the TPOT library. If not, see http://www.gnu.org/licenses/.
'''

import_relations = {
    '_variance_threshold':  {'sklearn.feature_selection': ['VarianceThreshold']},
    '_select_kbest':        {'sklearn.feature_selection': ['SelectKBest', 'f_classif']},
    '_select_fwe':          {'sklearn.feature_selection': ['SelectFwe', 'f_classif']},
    '_select_percentile':   {'sklearn.feature_selection': ['SelectPercentile', 'f_classif']},
    '_rfe':                 {'sklearn.feature_selection': ['RFE'], 'sklearn.svm': ['SVC']},
    '_standard_scaler':     {'sklearn.preprocessing': ['StandardScaler']},
    '_robust_scaler':       {'sklearn.preprocessing': ['RobustScaler']},
    '_min_max_scaler':      {'sklearn.preprocessing': ['MinMaxScaler']},
    '_max_abs_scaler':      {'sklearn.preprocessing': ['MaxAbsScaler']},
    '_binarizer':           {'sklearn.preprocessing': ['Binarizer']},
    '_polynomial_features': {'sklearn.preprocessing': ['PolynomialFeatures']},
    '_pca':                 {'sklearn.decomposition': ['RandomizedPCA']},
    '_fast_ica':            {'sklearn.decomposition': ['FastICA']},
    '_rbf':                 {'sklearn.kernel_approximation': ['RBFSampler']},
    '_nystroem':            {'sklearn.kernel_approximation': ['Nystroem']},
    '_decision_tree':       {'sklearn.tree': ['DecisionTreeClassifier']},
    '_random_forest':       {'sklearn.ensemble': ['RandomForestClassifier']},
    '_ada_boost':           {'sklearn.ensemble': ['AdaBoostClassifier']},
    '_extra_trees':         {'sklearn.ensemble': ['ExtraTreesClassifier']},
    '_gradient_boosting':   {'sklearn.ensemble': ['GradientBoostingClassifier']},
    '_logistic_regression': {'sklearn.linear_model': ['LogisticRegression']},
    '_passive_aggressive':  {'sklearn.linear_model': ['PassiveAggressiveClassifier']},
    '_svc':                 {'sklearn.svm': ['SVC']},
    '_linear_svc':          {'sklearn.svm': ['LinearSVC']},
    '_knnc':                {'sklearn.neighbors': ['KNeighborsClassifier']},
    '_feat_agg':            {'sklearn.cluster': ['FeatureAgglomeration']},
    '_gaussian_nb':         {'sklearn.naive_bayes': ['GaussianNB']},
    '_multinomial_nb':      {'sklearn.naive_bayes': ['MultinomialNB']},
    '_bernoulli_nb':        {'sklearn.naive_bayes': ['BernoulliNB']}
}

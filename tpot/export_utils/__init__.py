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

import deap

def replace_mathematical_operators(exported_pipeline):
    """Replace all of the mathematical operators with their results for use in export(self, output_file_name)

    Parameters
    ----------
    exported_pipeline:
       The current optimized pipeline

    Returns
    -------
    exported_pipeline:
       The current optimized pipeline after replacing the mathematical operators

    """
    while True:
        for i in range(len(exported_pipeline) - 1, -1, -1):
            node = exported_pipeline[i]
            if type(node) is deap.gp.Primitive and node.name in ['add', 'sub', 'mul', '_div']:
                val1 = int(exported_pipeline[i + 1].name)
                val2 = int(exported_pipeline[i + 2].name)
                if node.name == 'add':
                    new_val = val1 + val2
                elif node.name == 'sub':
                    new_val = val1 - val2
                elif node.name == 'mul':
                    new_val = val1 * val2
                else:
                    if val2 == 0:
                        new_val = 0
                    else:
                        new_val = float(val1) / float(val2)

                new_val = deap.gp.Terminal(symbolic=new_val, terminal=new_val, ret=new_val)
                exported_pipeline = exported_pipeline[:i] + [new_val] + exported_pipeline[i + 3:]
                break
        else:
            break

    return exported_pipeline

def unroll_nested_fuction_calls(exported_pipeline):
    """Unroll the nested function calls into serial code for use in TPOT.export()

    Parameters
    ----------
    exported_pipeline:
       The current optimized pipeline

    Returns
    -------
    exported_pipeline:
       The current optimized pipeline after unrolling the nested function calls
    pipeline_list:
       List of operators in the current optimized pipeline

    """
    pipeline_list = []
    result_num = 1
    while True:
        for node_index in range(len(exported_pipeline) - 1, -1, -1):
            node = exported_pipeline[node_index]
            if type(node) is not deap.gp.Primitive:
                continue

            node_params = exported_pipeline[node_index + 1:node_index + node.arity + 1]

            new_val = 'result{}'.format(result_num)
            operator_list = [new_val, node.name]
            operator_list.extend([x.name for x in node_params])
            pipeline_list.append(operator_list)
            result_num += 1
            new_val = deap.gp.Terminal(symbolic=new_val, terminal=new_val, ret=new_val)
            exported_pipeline = exported_pipeline[:node_index] + [new_val] + exported_pipeline[node_index + node.arity + 1:]
            break
        else:
            break
    return exported_pipeline, pipeline_list

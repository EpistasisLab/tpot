# -*- coding: utf-8 -*-

"""
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
"""

# Utility functions that convert the current optimized pipeline into its corresponding Python code
# For usage, see export() function in tpot.py

import deap


def unroll_nested_fuction_calls(exported_pipeline):
    """Unroll the nested function calls into serial code for use in TPOT.export()

    Parameters
    ----------
    exported_pipeline: deap.creator.Individual
       The current optimized pipeline

    Returns
    -------
    exported_pipeline: deap.creator.Individual
       The current optimized pipeline after unrolling the nested function calls
    pipeline_list: List
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

    # Replace 'ARG0' with 'input_df'
    for index in range(len(pipeline_list)):
        pipeline_list[index] = [x if x != 'ARG0' else 'input_df' for x in pipeline_list[index]]

    return pipeline_list


def generate_import_code(pipeline_list):
    """Generate all library import calls for use in TPOT.export()

    Parameters
    ----------
    pipeline_list: List
       List of operators in the current optimized pipeline

    Returns
    -------
    pipeline_text: String
       The Python code that imports all required library used in the current optimized pipeline

    """
    # operator[1] is the name of the operator
    operators_used = set([operator[1] for operator in pipeline_list])

    pipeline_text = 'import numpy as np\n'
    pipeline_text += 'import pandas as pd\n\n'

    # Always start with train_test_split as an import requirement
    pipeline_imports = {'sklearn.cross_validation': ['train_test_split']}

    # Operator names and imports required.
    # Dict structure:
    # {
    #   'operator_function':    {'module.to.import.from': ['ClassWithinModuleToImport']}
    # }
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
        '_linear_svc':          {'sklearn.svm': ['LinearSVC']},
        '_knnc':                {'sklearn.neighbors': ['KNeighborsClassifier']},
        '_feat_agg':            {'sklearn.cluster': ['FeatureAgglomeration']},
        '_gaussian_nb':         {'sklearn.naive_bayes': ['GaussianNB']},
        '_multinomial_nb':      {'sklearn.naive_bayes': ['MultinomialNB']},
        '_bernoulli_nb':        {'sklearn.naive_bayes': ['BernoulliNB']}
    }

    # Build import dict from operators used
    for op in operators_used:
        def merge_imports(old_dict, new_dict):
            # Key is a module name
            for key in new_dict.keys():
                if key in old_dict.keys():
                    # Append imports from the same module
                    old_dict[key] = set(list(old_dict[key]) + list(new_dict[key]))
                else:
                    old_dict[key] = new_dict[key]

        try:
            operator_import = import_relations[op]
            merge_imports(pipeline_imports, operator_import)
        except KeyError:
            pass  # Operator does not require imports

    # Build import string
    for key in sorted(pipeline_imports.keys()):
        module_list = ', '.join(sorted(pipeline_imports[key]))
        pipeline_text += 'from {} import {}\n'.format(key, module_list)

    pipeline_text += '''
# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)
'''

    return pipeline_text


def replace_function_calls(pipeline_list):
    """Replace the function calls with their corresponding Python code for use in TPOT.export()

    Parameters
    ----------
    pipeline_list:
       List of operators in the current optimized pipeline

    Returns
    -------
    operator_text:
       The Python code corresponding to the function calls in the current optimized pipeline

    """

    operator_text = ''
    for operator in pipeline_list:
        operator_num = int(operator[0].strip('result'))
        result_name = operator[0]
        operator_name = operator[1]

        # Make copies of the data set for each reference to input_df
        if operator[2] == 'input_df':
            operator[2] = 'result{}'.format(operator_num)
            operator_text += '\n{} = tpot_data.copy()\n'.format(operator[2])

        if len(operator) > 3 and operator[3] == 'input_df':
            operator[3] = 'result{}'.format(operator_num)
            operator_text += '\n{} = tpot_data.copy()\n'.format(operator[3])

        # Replace the TPOT functions with their corresponding Python code
        if operator_name == '_decision_tree':
            min_weight = float(operator[3])

            operator_text += '\n# Perform classification with a decision tree classifier'
            operator_text += ('\ndtc{OPERATOR_NUM} = DecisionTreeClassifier('
                              'min_weight_fraction_leaf={MIN_WEIGHT})\n').format(OPERATOR_NUM=operator_num,
                                                                                MIN_WEIGHT=min_weight)
            operator_text += ('''dtc{OPERATOR_NUM}.fit({INPUT_DF}.loc[training_indices].drop('class', '''
                              '''axis=1).values, {INPUT_DF}.loc[training_indices, 'class'].values)\n''').format(OPERATOR_NUM=operator_num,
                                                                                                                INPUT_DF=operator[2])
            if result_name != operator[2]:
                operator_text += '{OUTPUT_DF} = {INPUT_DF}.copy()\n'.format(OUTPUT_DF=result_name, INPUT_DF=operator[2])
            operator_text += ('''{OUTPUT_DF}['dtc{OPERATOR_NUM}-classification'] = dtc{OPERATOR_NUM}.predict('''
                              '''{OUTPUT_DF}.drop('class', axis=1).values)\n''').format(OUTPUT_DF=result_name,
                                                                                        OPERATOR_NUM=operator_num)

        elif operator_name == '_random_forest':
            min_weight = min(0.5, max(0., operator[3]))

            operator_text += '\n# Perform classification with a random forest classifier'
            operator_text += ('\nrfc{OPERATOR_NUM} = RandomForestClassifier('
                              'n_estimators=500, min_weight_fraction_leaf={MIN_WEIGHT})\n').format(OPERATOR_NUM=operator_num, MIN_WEIGHT=min_weight)
            operator_text += ('''rfc{OPERATOR_NUM}.fit({INPUT_DF}.loc[training_indices].drop('class', axis=1).values, '''
                              '''{INPUT_DF}.loc[training_indices, 'class'].values)\n''').format(OPERATOR_NUM=operator_num,
                                                                                                INPUT_DF=operator[2])
            if result_name != operator[2]:
                operator_text += '{OUTPUT_DF} = {INPUT_DF}.copy()\n'.format(OUTPUT_DF=result_name, INPUT_DF=operator[2])
            operator_text += ('''{OUTPUT_DF}['rfc{OPERATOR_NUM}-classification'] = '''
                              '''rfc{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)\n''').format(OUTPUT_DF=result_name,
                                                                                                                  OPERATOR_NUM=operator_num)

        elif operator_name == '_logistic_regression':
            C = min(50., max(0.0001, float(operator[3])))

            penalty_values = ['l1', 'l2']
            penalty_selection = penalty_values[int(operator[4]) % len(penalty_values)]

            dual = bool(operator[5])

            if penalty_selection == 'l1':
                dual = False

            operator_text += '\n# Perform classification with a logistic regression classifier'
            operator_text += '\nlrc{OPERATOR_NUM} = LogisticRegression(C={C}, dual={DUAL}, penalty="{PENALTY}")\n'.format(OPERATOR_NUM=operator_num,
                                                                                                                        C=C,
                                                                                                                        PENALTY=penalty_selection,
                                                                                                                        DUAL=dual)
            operator_text += ('''lrc{OPERATOR_NUM}.fit({INPUT_DF}.loc[training_indices].drop('class', axis=1).values, '''
                              '''{INPUT_DF}.loc[training_indices, 'class'].values)\n''').format(OPERATOR_NUM=operator_num, INPUT_DF=operator[2])
            if result_name != operator[2]:
                operator_text += '{OUTPUT_DF} = {INPUT_DF}.copy()\n'.format(OUTPUT_DF=result_name, INPUT_DF=operator[2])
            operator_text += ('''{OUTPUT_DF}['lrc{OPERATOR_NUM}-classification'] = lrc{OPERATOR_NUM}.predict('''
                              '''{OUTPUT_DF}.drop('class', axis=1).values)\n''').format(OUTPUT_DF=result_name,
                                                                                        OPERATOR_NUM=operator_num)

        elif operator_name == '_knnc':
            n_neighbors = int(operator[3])
            if n_neighbors < 2:
                n_neighbors = 2
            else:
                n_neighbors = 'min({N_NEIGHBORS}, len(training_indices))'.format(N_NEIGHBORS=n_neighbors)

            weights_values = ['uniform', 'distance']
            weights_selection = weights_values[int(operator[4]) % len(weights_values)]

            operator_text += '\n# Perform classification with a k-nearest neighbor classifier'
            operator_text += '\nknnc{OPERATOR_NUM} = KNeighborsClassifier(n_neighbors={N_NEIGHBORS}, weights={WEIGHTS})\n'.format(OPERATOR_NUM=operator_num,
                                                                                                               N_NEIGHBORS=n_neighbors,
                                                                                                               WEIGHTS=weights_selection)
            operator_text += ('''knnc{OPERATOR_NUM}.fit({INPUT_DF}.loc[training_indices].drop('class', axis=1).values, '''
                              '''{INPUT_DF}.loc[training_indices, 'class'].values)\n''').format(OPERATOR_NUM=operator_num,
                                                                                                INPUT_DF=operator[2])
            if result_name != operator[2]:
                operator_text += '{OUTPUT_DF} = {INPUT_DF}.copy()\n'.format(OUTPUT_DF=result_name, INPUT_DF=operator[2])
            operator_text += ('''{OUTPUT_DF}['knnc{OPERATOR_NUM}-classification'] = knnc{OPERATOR_NUM}.predict('''
                              '''{OUTPUT_DF}.drop('class', axis=1).values)\n''').format(OUTPUT_DF=result_name,
                                                                                        OPERATOR_NUM=operator_num)

        elif operator_name == '_ada_boost':
            learning_rate = min(1., max(0.0001, float(operator[3])))
            n_estimators = 500

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with an Ada Boost classifier
adab{OPERATOR_NUM} = AdaBoostClassifier(learning_rate={LEARNING_RATE}, n_estimators={N_ESTIMATORS}, random_state=42)
adab{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['adab{OPERATOR_NUM}-classification'] = adab{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, N_ESTIMATORS=n_estimators, LEARNING_RATE=learning_rate)

        elif operator_name == '_bernoulli_nb':
            alpha = float(operator[3])
            binarize = float(operator[4])

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with a BernoulliNB classifier
bnb{OPERATOR_NUM} = BernoulliNB(alpha={ALPHA}, binarize={BINARIZE})
bnb{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['bnb{OPERATOR_NUM}-classification'] = bnb{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, ALPHA=alpha, BINARIZE=binarize)

        elif operator_name == '_extra_trees':
            criterion = int(operator[3])
            max_features = min(1., max(0., float(operator[4])))
            min_weight = min(0.5, max(0., float(operator[5])))

            criterion_values = ['gini', 'entropy']
            criterion_selection = criterion_values[criterion % len(criterion_values)]

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

# Perform classification with an extra trees classifier
etc{OPERATOR_NUM} = ExtraTreesClassifier(criterion="{CRITERION}", max_features={MAX_FEATURES}, min_weight_fraction_leaf={MIN_WEIGHT}, n_estimators=500, random_state=42)
etc{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['etc{OPERATOR_NUM}-classification'] = etc{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, INPUT_DF=operator[2], CRITERION=criterion_selection, MAX_FEATURES=max_features, MIN_WEIGHT=min_weight)

        elif operator_name == '_gaussian_nb':
            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with a gaussian naive bayes classifier
gnb{OPERATOR_NUM} = GaussianNB()
gnb{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['gnb{OPERATOR_NUM}-classification'] = gnb{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num)

        elif operator_name == '_multinomial_nb':
            alpha = float(operator[3])

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Performan classification with a multinomial naive bayes classifier
mnb{OPERATOR_NUM} = MultinomialNB(alpha={ALPHA}, fit_prior=True)
mnb{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['mnb{OPERATOR_NUM}-classification'] = mnb{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, ALPHA=alpha)

        elif operator_name == '_linear_svc':
            penalty_values = ['l1', 'l2']
            penalty_selection = penalty_values[int(operator[4]) % len(penalty_values)]

            C = min(50., max(0.0001, operator[3]))

            dual = bool(operator[5])

            if penalty_selection == 'l1':
                dual = False

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with a LinearSVC classifier
lsvc{OPERATOR_NUM} = LinearSVC(C={C}, penalty="{PENALTY}", dual={DUAL}, random_state=42)
lsvc{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['lsvc{OPERATOR_NUM}-classification'] = lsvc{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, C=C, PENALTY=penalty_selection, DUAL=dual)

        elif operator_name == '_passive_aggressive':
            C = min(1., max(0.0001, float(operator[3])))
            loss = int(operator[4])

            loss_values = ['hinge', 'squared_hinge']
            loss_selection = loss_values[loss % len(loss_values)]

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with a passive aggressive classifier
pagr{OPERATOR_NUM} = PassiveAggressiveClassifier(C={C}, loss="{LOSS}", fit_intercept=True, random_state=42)
pagr{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['pagr{OPERATOR_NUM}-classification'] = pagr{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, C=C, LOSS=loss_selection)

        elif operator_name == '_gradient_boosting':
            learning_rate = min(1., max(float(operator[3]), 0.0001))
            max_features = min(1., max(0., float(operator[4])))
            min_weight = min(0.5, max(0., float(operator[5])))

            if result_name != operator[2]:
                operator_text += "\n{OUTPUT_DF} = {INPUT_DF}.copy()".format(OUTPUT_DF=result_name, INPUT_DF=operator[2])

            operator_text += """
# Perform classification with a gradient boosting classifier
gbc{OPERATOR_NUM} = GradientBoostingClassifier(learning_rate={LEARNING_RATE}, max_features={MAX_FEATURES}, min_weight_fraction_leaf={MIN_WEIGHT}, n_estimators=500, random_state=42)
gbc{OPERATOR_NUM}.fit({OUTPUT_DF}.loc[training_indices].drop('class', axis=1).values, {OUTPUT_DF}.loc[training_indices, 'class'].values)

{OUTPUT_DF}['gbc{OPERATOR_NUM}-classification'] = gbc{OPERATOR_NUM}.predict({OUTPUT_DF}.drop('class', axis=1).values)
""".format(OUTPUT_DF=result_name, OPERATOR_NUM=operator_num, LEARNING_RATE=learning_rate, MAX_FEATURES=max_features, MIN_WEIGHT=min_weight)

        elif operator_name == '_combine_dfs':
            operator_text += '\n# Combine two DataFrames'
            operator_text += ('\n{OUTPUT_DF} = {INPUT_DF1}.join({INPUT_DF2}[[column for column in {INPUT_DF2}.columns.values '
                              'if column not in {INPUT_DF1}.columns.values]])\n').format(INPUT_DF1=operator[2],
                                                                                         INPUT_DF2=operator[3],
                                                                                         OUTPUT_DF=result_name)

        elif operator_name == '_variance_threshold':
            threshold = float(operator[3])

            operator_text += '''
# Use Scikit-learn's VarianceThreshold for feature selection
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

selector = VarianceThreshold(threshold={THRESHOLD})
try:
    selector.fit(training_features.values)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {OUTPUT_DF} = {INPUT_DF}[mask_cols]
except ValueError:
    # None of the features meet the variance threshold
    {OUTPUT_DF} = {INPUT_DF}[['class']]
'''.format(INPUT_DF=operator[2], THRESHOLD=threshold, OUTPUT_DF=result_name)

        elif operator_name == '_select_kbest':
            k = int(operator[3])
            if k < 1:
                k = 1
            k = 'min({K}, len(training_features.columns))'.format(K=k)

            operator_text += '''
# Use Scikit-learn's SelectKBest for feature selection
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)
training_class_vals = {INPUT_DF}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {OUTPUT_DF} = {INPUT_DF}.copy()
else:
    selector = SelectKBest(f_classif, k={K})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {OUTPUT_DF} = {INPUT_DF}[mask_cols]
'''.format(INPUT_DF=operator[2], K=k, OUTPUT_DF=result_name)

        # SelectFwe based on the SelectKBest code
        elif operator_name == '_select_fwe':
            alpha = float(operator[3])
            if alpha > 0.05:
                alpha = 0.05
            elif alpha <= 0.001:
                alpha = 0.001
            operator_text += '''
training_features = {INPUT_DF}.loc[training_indices].drop(['class'], axis=1)
training_class_vals = {INPUT_DF}.loc[training_indices, 'class'].values
if len(training_features.columns.values) == 0:
    {OUTPUT_DF} = {INPUT_DF}.copy()
else:
    selector = SelectFwe(f_classif, alpha={ALPHA})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {OUTPUT_DF} = {INPUT_DF}[mask_cols]
'''.format(INPUT_DF=operator[2], ALPHA=alpha, OUTPUT_DF=result_name)

        elif operator_name == '_select_percentile':
            percentile = int(operator[3])

            if percentile < 0:
                percentile = 0
            elif percentile > 100:
                percentile = 100

            operator_text += '''
# Use Scikit-learn's SelectPercentile for feature selection
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)
training_class_vals = {INPUT_DF}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {OUTPUT_DF} = {INPUT_DF}.copy()
else:
    selector = SelectPercentile(f_classif, percentile={PERCENTILE})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {OUTPUT_DF} = {INPUT_DF}[mask_cols]
'''.format(INPUT_DF=operator[2], PERCENTILE=percentile, OUTPUT_DF=result_name)

        elif operator_name == '_rfe':
            n_features_to_select = int(operator[3])
            step = float(operator[4])

            if n_features_to_select < 1:
                n_features_to_select = 1
            n_features_to_select = 'min({N_FEATURES_TO_SELECT}, len(training_features.columns))'.format(N_FEATURES_TO_SELECT=n_features_to_select)

            if step < 0.1:
                step = 0.1
            elif step >= 1.:
                step = 0.99

            operator_text += '''
# Use Scikit-learn's Recursive Feature Elimination (RFE) for feature selection
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)
training_class_vals = {INPUT_DF}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {OUTPUT_DF} = {INPUT_DF}.copy()
else:
    selector = RFE(SVC(kernel='linear'), n_features_to_select={N_FEATURES_TO_SELECT}, step={STEP})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {OUTPUT_DF} = {INPUT_DF}[mask_cols]
'''.format(INPUT_DF=operator[2], N_FEATURES_TO_SELECT=n_features_to_select, STEP=step, OUTPUT_DF=result_name)

        elif operator_name == '_standard_scaler':
            operator_text += '''
# Use Scikit-learn's StandardScaler to scale the features
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = StandardScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=scaled_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

        elif operator_name == '_robust_scaler':
            operator_text += '''
# Use Scikit-learn's RobustScaler to scale the features
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = RobustScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=scaled_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

        elif operator_name == '_min_max_scaler':
            operator_text += '''
# Use Scikit-learn's MinMaxScaler to scale the features
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = MinMaxScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=scaled_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

        elif operator_name == '_max_abs_scaler':
            operator_text += '''
# Use Scikit-learn's MaxAbsScaler to scale the features
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = MaxAbsScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=scaled_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

        elif operator_name == '_binarizer':
            threshold = float(operator[3])
            operator_text += '''
# Use Scikit-learn's Binarizer to scale the features
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    scaler = Binarizer(threshold={THRESHOLD})
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=scaled_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], THRESHOLD=threshold, OUTPUT_DF=result_name)

        elif operator_name == '_polynomial_features':
            operator_text += '''
# Use Scikit-learn's PolynomialFeatures to construct new features from the existing feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0 and len(training_features.columns.values) <= 700:
    # The feature constructor must be fit on only the training data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(training_features.values.astype(np.float64))
    constructed_features = poly.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=constructed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

        elif operator_name == '_pca':
            iterated_power = int(operator[3])

            if iterated_power < 1:
                iterated_power = 1
            elif iterated_power > 10:
                iterated_power = 10

            operator_text += '''
# Use Scikit-learn's RandomizedPCA to transform the feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # PCA must be fit on only the training data
    pca = RandomizedPCA(iterated_power={ITERATED_POWER})
    pca.fit(training_features.values.astype(np.float64))
    transformed_features = pca.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=transformed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], ITERATED_POWER=iterated_power, OUTPUT_DF=result_name)

        elif operator_name == '_rbf':
            gamma = float(operator[3])

            operator_text += '''
# Use Scikit-learn's RBFSampler to transform the feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # RBF must be fit on only the training data
    rbf = RBFSampler(gamma={GAMMA})
    rbf.fit(training_features.values.astype(np.float64))
    transformed_features = rbf.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=transformed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], GAMMA=gamma, OUTPUT_DF=result_name)

        elif operator_name == '_fast_ica':
            tol = max(float(operator[3]), 0.0001)  # Ensure tol is not too small

            operator_text += '''
# Use Scikit-learn's FastICA to transform the feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # FastICA must be fit on only the training data
    ica = FastICA(tol={TOL}, random_state=42)
    ica.fit(training_features.values.astype(np.float64))
    transformed_features = ica.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=transformed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], TOL=tol, OUTPUT_DF=result_name)

        elif operator_name == '_feat_agg':
            n_clusters = min(1, int(operator[3]))
            affinity = int(operator[4])
            linkage = int(operator[5])

            affinity_types = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
            linkage_types = ['ward', 'complete', 'average']

            linkage_name = linkage_types[linkage % len(linkage_types)]

            if linkage_name == 'ward':
                affinity_name = 'euclidean'
            else:
                affinity_name = affinity_types[affinity % len(affinity_types)]

            operator_text += '''
# Use Scikit-learn's FeatureAgglomeration to transform the feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # FeatureAgglomeration must be fit on only the training data
    fa = FeatureAgglomeration(n_clusters={N_CLUSTERS}, affinity='{AFFINITY}', linkage='{LINKAGE}')
    fa.fit(training_features.values.astype(np.float64))
    transformed_features = fa.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=transformed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], N_CLUSTERS=n_clusters, AFFINITY=affinity_name, LINKAGE=linkage_name, OUTPUT_DF=result_name)

        elif operator_name == '_nystroem':
            kernel = int(operator[3])
            gamma = float(operator[4])
            n_components = int(operator[5])

            # Kernel functions from sklearn.metrics.pairwise
            kernel_types = ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid']
            kernel_name = kernel_types[kernel % len(kernel_types)]

            if n_components < 1:
                n_components = 1
            else:
                n_components = 'min({}, len(training_features.columns.values))'.format(n_components)

            operator_text += '''
# Use Scikit-learn's Nystroem to transform the feature set
training_features = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # FeatureAgglomeration must be fit on only the training data
    nys = Nystroem(kernel='{KERNEL}', gamma={GAMMA}, n_components={N_COMPONENTS})
    nys.fit(training_features.values.astype(np.float64))
    transformed_features = nys.transform({INPUT_DF}.drop('class', axis=1).values.astype(np.float64))
    {OUTPUT_DF} = pd.DataFrame(data=transformed_features)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], KERNEL=kernel_name, GAMMA=gamma, N_COMPONENTS=n_components, OUTPUT_DF=result_name)

        elif operator_name == '_zero_count':
            operator_text += '''
# Add virtual features for number of zeros and non-zeros per row
feature_cols_only = {INPUT_DF}.loc[training_indices].drop('class', axis=1)

if len(feature_cols_only.columns.values) > 0:
    non_zero_col = np.array([np.count_nonzero(row) for i, row in {INPUT_DF}.iterrows()]).astype(np.float64)
    zero_col     = np.array([(len(feature_cols_only.columns.values) - x) for x in non_zero_col]).astype(np.float64)

    {OUTPUT_DF} = {INPUT_DF}.copy()
    {OUTPUT_DF}['non_zero'] = pd.Series(non_zero_col, index={OUTPUT_DF}.index)
    {OUTPUT_DF}['zero_col'] = pd.Series(zero_col, index={OUTPUT_DF}.index)
    {OUTPUT_DF}['class'] = {INPUT_DF}['class'].values
else:
    {OUTPUT_DF} = {INPUT_DF}.copy()
'''.format(INPUT_DF=operator[2], OUTPUT_DF=result_name)

    return operator_text

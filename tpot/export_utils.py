# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

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

import deap


def get_by_name(opname, operators):
    """Return operator class instance by name.

    Parameters
    ----------
    opname: str
        Name of the sklearn class that belongs to a TPOT operator
    operators: list
        List of operator classes from operator library

    Returns
    -------
    ret_op_class: class
        An operator class

    """
    ret_op_classes = [op for op in operators if op.__name__ == opname]

    if len(ret_op_classes) == 0:
        raise TypeError('Cannot found operator {} in operator dictionary'.format(opname))
    elif len(ret_op_classes) > 1:
        raise ValueError(
            'Found duplicate operators {} in operator dictionary. Please check '
            'your dictionary file.'.format(opname)
        )
    ret_op_class = ret_op_classes[0]
    return ret_op_class


def export_pipeline(exported_pipeline, operators, pset, impute=False, pipeline_score=None, random_state=None):
    """Generate source code for a TPOT Pipeline.

    Parameters
    ----------
    exported_pipeline: deap.creator.Individual
        The pipeline that is being exported
    operators:
        List of operator classes from operator library
    pipeline_score:
        Optional pipeline score to be saved to the exported file

    Returns
    -------
    pipeline_text: str
        The source code representing the pipeline

    """
    # Unroll the nested function calls into serial code
    pipeline_tree = expr_to_tree(exported_pipeline, pset)

    # Have the exported code import all of the necessary modules and functions
    pipeline_text = generate_import_code(exported_pipeline, operators, impute)

    pipeline_code = pipeline_code_wrapper(generate_export_pipeline_code(pipeline_tree, operators))

    if pipeline_code.count("FunctionTransformer(copy)"):
        pipeline_text += """from sklearn.preprocessing import FunctionTransformer
from copy import copy
"""

    pipeline_text += """
# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \\
            train_test_split(features, tpot_data['target'].values, random_state={})
""".format(random_state)

    # Add the imputation step if it was used by TPOT
    if impute:
        pipeline_text += """
imputer = Imputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)
"""

    if pipeline_score is not None:
        pipeline_text += '\n# Average CV score on the training set was:{}'.format(pipeline_score)
    pipeline_text += '\n'

    # Replace the function calls with their corresponding Python code
    pipeline_text += pipeline_code

    return pipeline_text


def expr_to_tree(ind, pset):
    """Convert the unstructured DEAP pipeline into a tree data-structure.

    Parameters
    ----------
    ind: deap.creator.Individual
       The pipeline that is being exported

    Returns
    -------
    pipeline_tree: list
       List of operators in the current optimized pipeline

    EXAMPLE:
        pipeline:
            "DecisionTreeClassifier(input_matrix, 28.0)"
        pipeline_tree:
            ['DecisionTreeClassifier', 'input_matrix', 28.0]

    """
    def prim_to_list(prim, args):
        if isinstance(prim, deap.gp.Terminal):
            if prim.name in pset.context:
                return pset.context[prim.name]
            else:
                return prim.value

        return [prim.name] + args

    tree = []
    stack = []
    for node in ind:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            tree = prim_to_list(prim, args)
            if len(stack) == 0:
                break   # If stack is empty, all nodes should have been seen
            stack[-1][1].append(tree)

    return tree


def generate_import_code(pipeline, operators, impute=False):
    """Generate all library import calls for use in TPOT.export().

    Parameters
    ----------
    pipeline: List
        List of operators in the current optimized pipeline
    operators:
        List of operator class from operator library
    impute : bool
        Whether to impute new values in the feature set.

    Returns
    -------
    pipeline_text: String
        The Python code that imports all required library used in the current
        optimized pipeline

    """
    def merge_imports(old_dict, new_dict):
        # Key is a module name
        for key in new_dict.keys():
            if key in old_dict.keys():
                # Union imports from the same module
                old_dict[key] = set(old_dict[key]) | set(new_dict[key])
            else:
                old_dict[key] = set(new_dict[key])

    operators_used = [x.name for x in pipeline if isinstance(x, deap.gp.Primitive)]
    pipeline_text = 'import numpy as np\nimport pandas as pd\n'
    pipeline_imports = _starting_imports(operators, operators_used)

    # Build dict of import requirments from list of operators
    import_relations = {op.__name__: op.import_hash for op in operators}

    # Add the imputer if necessary
    if impute:
        pipeline_imports['sklearn.preprocessing'] = ['Imputer']

    # Build import dict from operators used
    for op in operators_used:
        try:
            operator_import = import_relations[op]
            merge_imports(pipeline_imports, operator_import)
        except KeyError:
            pass  # Operator does not require imports

    # Build import string
    for key in sorted(pipeline_imports.keys()):
        module_list = ', '.join(sorted(pipeline_imports[key]))
        pipeline_text += 'from {} import {}\n'.format(key, module_list)

    return pipeline_text


def _starting_imports(operators, operators_used):
    # number of operators
    num_op = len(operators_used)

    # number of classifier/regressor or CombineDFs
    num_op_root = 0
    for op in operators_used:
        if op != 'CombineDFs':
            tpot_op = get_by_name(op, operators)
            if tpot_op.root:
                num_op_root += 1
        else:
            num_op_root += 1

    if num_op_root > 1:
        return {
            'sklearn.model_selection':  ['train_test_split'],
            'sklearn.pipeline':         ['make_pipeline', 'make_union'],
            'tpot.builtins':  ['StackingEstimator'],
        }
    elif num_op > 1:
        return {
            'sklearn.model_selection':  ['train_test_split'],
            'sklearn.pipeline':         ['make_pipeline']
        }
    # if operators # == 1 and classifier/regressor # == 1, this import statement is simpler
    else:
        return {
            'sklearn.model_selection':  ['train_test_split']
        }


def pipeline_code_wrapper(pipeline_code):
    """Generate code specific to the execution of the sklearn pipeline.

    Parameters
    ----------
    pipeline_code: str
        Code that defines the final sklearn pipeline

    Returns
    -------
    Source code for the sklearn pipeline and calls to fit and predict

    """
    return """exported_pipeline = {}

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
""".format(pipeline_code)


def generate_pipeline_code(pipeline_tree, operators):
    """Generate code specific to the construction of the sklearn Pipeline.

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = _process_operator(pipeline_tree, operators)
    pipeline_text = "make_pipeline(\n{STEPS}\n)".format(STEPS=_indent(",\n".join(steps), 4))
    return pipeline_text


def generate_export_pipeline_code(pipeline_tree, operators):
    """Generate code specific to the construction of the sklearn Pipeline for export_pipeline.

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = _process_operator(pipeline_tree, operators)
    # number of steps in a pipeline
    num_step = len(steps)
    if num_step > 1:
        pipeline_text = "make_pipeline(\n{STEPS}\n)".format(STEPS=_indent(",\n".join(steps), 4))
    # only one operator (root = True)
    else:
        pipeline_text = "{STEPS}".format(STEPS=_indent(",\n".join(steps), 0))

    return pipeline_text


def _process_operator(operator, operators, depth=0):
    steps = []
    op_name = operator[0]

    if op_name == "CombineDFs":
        steps.append(
            _combine_dfs(operator[1], operator[2], operators)
        )
    else:
        input_name, args = operator[1], operator[2:]
        tpot_op = get_by_name(op_name, operators)

        if input_name != 'input_matrix':
            steps.extend(_process_operator(input_name, operators, depth + 1))

        # If the step is an estimator and is not the last step then we must
        # add its guess as synthetic feature(s)
        # classification prediction for both regression and classification
        # classification probabilities for classification if available
        if tpot_op.root and depth > 0:
            steps.append(
                "StackingEstimator(estimator={})".
                format(tpot_op.export(*args))
            )
        else:
            steps.append(tpot_op.export(*args))
    return steps


def _indent(text, amount):
    """Indent a multiline string by some number of spaces.

    Parameters
    ----------
    text: str
        The text to be indented
    amount: int
        The number of spaces to indent the text

    Returns
    -------
    indented_text

    """
    indentation = amount * ' '
    return indentation + ('\n' + indentation).join(text.split('\n'))


def _combine_dfs(left, right, operators):
    def _make_branch(branch):
        if branch == "input_matrix":
            return "FunctionTransformer(copy)"
        elif branch[0] == "CombineDFs":
            return _combine_dfs(branch[1], branch[2], operators)
        elif branch[1] == "input_matrix":  # If depth of branch == 1
            tpot_op = get_by_name(branch[0], operators)

            if tpot_op.root:
                return "StackingEstimator(estimator={})".format(_process_operator(branch, operators)[0])
            else:
                return _process_operator(branch, operators)[0]
        else:  # We're going to have to make a pipeline
            tpot_op = get_by_name(branch[0], operators)

            if tpot_op.root:
                return "StackingEstimator(estimator={})".format(generate_pipeline_code(branch, operators))
            else:
                return generate_pipeline_code(branch, operators)

    return "make_union(\n{},\n{}\n)".\
        format(_indent(_make_branch(left), 4), _indent(_make_branch(right), 4))

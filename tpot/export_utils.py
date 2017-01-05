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
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along
with the TPOT library. If not, see http://www.gnu.org/licenses/.

"""

import deap
from . import operators


def export_pipeline(exported_pipeline):
    """Generates the source code of a TPOT Pipeline

    Parameters
    ----------
    exported_pipeline: deap.creator.Individual
        The pipeline that is being exported

    Returns
    -------
    pipeline_text: str
        The source code representing the pipeline

    """
    # Unroll the nested function calls into serial code
    pipeline_tree = expr_to_tree(exported_pipeline)

    # Have the exported code import all of the necessary modules and functions
    pipeline_text = generate_import_code(exported_pipeline)

    # Replace the function calls with their corresponding Python code
    pipeline_text += pipeline_code_wrapper(generate_pipeline_code(pipeline_tree))

    return pipeline_text


def expr_to_tree(ind):
    """Convert the unstructured DEAP pipeline into a tree data-structure

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


def generate_import_code(pipeline):
    """Generate all library import calls for use in TPOT.export()

    Parameters
    ----------
    pipeline: List
       List of operators in the current optimized pipeline

    Returns
    -------
    pipeline_text: String
       The Python code that imports all required library used in the current
       optimized pipeline

    """
    # operator[1] is the name of the operator
    operators_used = [x.name for x in pipeline if isinstance(x, deap.gp.Primitive)]

    pipeline_text = 'import numpy as np\n\n'

    # Always start with these imports
    pipeline_imports = {
        'sklearn.model_selection':  ['train_test_split'],
        'sklearn.pipeline':         ['make_pipeline', 'make_union'],
        'sklearn.preprocessing':    ['FunctionTransformer'],
        'sklearn.ensemble':         ['VotingClassifier']
    }

    # Build dict of import requirments from list of operators
    import_relations = {}
    for op in operators.Operator.inheritors():
        import_relations[op.__name__] = op.import_hash

    # Build import dict from operators used
    for op in operators_used:
        def merge_imports(old_dict, new_dict):
            # Key is a module name
            for key in new_dict.keys():
                if key in old_dict.keys():
                    # Union imports from the same module
                    old_dict[key] = set(old_dict[key]) | set(new_dict[key])
                else:
                    old_dict[key] = set(new_dict[key])

        try:
            operator_import = import_relations[op]
            merge_imports(pipeline_imports, operator_import)
        except KeyError:
            pass  # Operator does not require imports

    # Build import string
    for key in sorted(pipeline_imports.keys()):
        module_list = ', '.join(sorted(pipeline_imports[key]))
        pipeline_text += 'from {} import {}\n'.format(key, module_list)

    pipeline_text += """
# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \\
    train_test_split(features, tpot_data['class'], random_state=42)
"""

    return pipeline_text


def pipeline_code_wrapper(pipeline_code):
    """Generate code specific to the execution of the sklearn pipeline

    Parameters
    ----------
    pipeline_code: str
        Code that defines the final sklearn pipeline

    Returns
    -------
    Source code for the sklearn pipeline and calls to fit and predict

    """
    return """
exported_pipeline = {}

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
""".format(pipeline_code)


def generate_pipeline_code(pipeline_tree):
    """Generate code specific to the construction of the sklearn Pipeline

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = process_operator(pipeline_tree)
    pipeline_text = "make_pipeline(\n{STEPS}\n)".format(STEPS=_indent(",\n".join(steps), 4))
    return pipeline_text


def process_operator(operator, depth=0):
    steps = []
    op_name = operator[0]

    if op_name == "CombineDFs":
        steps.append(
            _combine_dfs(operator[1], operator[2])
        )
    else:
        input_name, args = operator[1], operator[2:]
        tpot_op = operators.Operator.get_by_name(op_name)

        if input_name != 'input_matrix':
            steps.extend(process_operator(input_name, depth + 1))

        # If the step is an estimator and is not the last step then we must
        # add its guess as a synthetic feature
        if tpot_op.root and depth > 0:
            steps.append(
                "make_union(VotingClassifier([(\"est\", {})]), FunctionTransformer(lambda X: X))".
                format(tpot_op.export(*args))
            )
        else:
            steps.append(tpot_op.export(*args))

    return steps


def _indent(text, amount):
    """Indent a multiline string by some number of spaces

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


def _combine_dfs(left, right):
    def _make_branch(branch):
        if branch == "input_matrix":
            return "FunctionTransformer(lambda X: X)"
        elif branch[0] == "CombineDFs":
            return _combine_dfs(branch[1], branch[2])
        elif branch[1] == "input_matrix":  # If depth of branch == 1
            tpot_op = operators.Operator.get_by_name(branch[0])

            if tpot_op.root:
                return """make_union(VotingClassifier([('branch',
{}
)]), FunctionTransformer(lambda X: X))""".format(_indent(process_operator(branch)[0], 4))
            else:
                return process_operator(branch)[0]
        else:  # We're going to have to make a pipeline
            tpot_op = operators.Operator.get_by_name(branch[0])

            if tpot_op.root:
                return """make_union(VotingClassifier([('branch',
{}
)]), FunctionTransformer(lambda X: X))""".format(_indent(generate_pipeline_code(branch), 4))
            else:
                return generate_pipeline_code(branch)

    return "make_union(\n{},\n{}\n)".\
        format(_indent(_make_branch(left), 4), _indent(_make_branch(right), 4))

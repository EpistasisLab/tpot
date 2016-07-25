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
    """Generates the source code of a Python script that recreates the
    functionality of a TPOT pipeline

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
    pipeline_tree = expr_to_tree(exported_pipeline)[0]

    # Have the exported code import all of the necessary modules and functions
    pipeline_text = generate_import_code(exported_pipeline)

    # Replace the function calls with their corresponding Python code
    pipeline_text += pipeline_code_wrapper(generate_pipeline_code(pipeline_tree))

    return pipeline_text


def expr_to_tree(pipeline):
    """Convert the unstructured DEAP pipeline into a tree data-structure

    Parameters
    ----------
    pipeline: deap.creator.Individual
       The pipeline that is being exported

    Returns
    -------
    pipeline_tree: list
       List of operators in the current optimized pipeline

    EXAMPLE:
        pipeline:
            "DecisionTreeClassifier(
                CombineDFs(
                    KNeighborsClassifier(
                        RBFSampler(
                            input_df,
                            0.9
                        ),
                    23,
                    35
                    ),
                    KNeighborsClassifier(
                        input_df,
                        23,
                        35
                    )
                ),
                28.0
            )"
        pipeline_tree:
            ['DecisionTreeClassifier',
                ['CombineDFs',
                    ['KNeighborsClassifier',
                        ['RBFSampler',
                            'input_df',
                            0.90000000000000002
                        ],
                        23,
                        35
                    ],
                    ['KNeighborsClassifier',
                        'input_df',
                        23,
                        35
                    ]
                ],
                28.0
            ]

    """
    pipeline_tree = []
    iterable = enumerate(pipeline)

    for i, node in iterable:
        if isinstance(node, deap.gp.Primitive):
            arity = _true_arity(pipeline[i:])
            primitive_args = expr_to_tree(pipeline[i + 1:i + 1 + arity])
            pipeline_tree.append([node.name, *primitive_args])

            # Skip past the primitive's args
            [next(iterable) for x in range(arity)]
        else:
            pipeline_tree.append(node.value)

    return pipeline_tree


def _true_arity(pipeline):
    """Recursively determines the number of atoms in a pipeline snip that are
    contained within the outermost primitive.

    Parameters
    ----------
    pipeline: list
        The partial pipeline to be evaulated

    Returns
    -------
    arity: int
        The number of atoms contained within the primitve

    """
    if len(pipeline) == 0:
        return 0

    arity = 0
    if pipeline[0].name == "CombineDFs":
        left_arity = _true_arity(pipeline[1:])
        right_arity = _true_arity(pipeline[1 + left_arity:])

        arity += left_arity + right_arity - 2
    if isinstance(pipeline[0], deap.gp.Primitive):
        arity += pipeline[0].arity + _true_arity(pipeline[1:])

    return arity


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

    pipeline_text = 'import numpy as np\n'
    pipeline_text += 'import pandas as pd\n\n'

    # Always start with these imports
    pipeline_imports = {
        'sklearn.cross_validation': ['train_test_split'],
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
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify=tpot_data['class'].values, train_size=0.75, test_size=0.25)
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

exported_pipeline.fit(tpot_data.loc[training_indices].drop('class', axis=1).values,
                      tpot_data.loc[training_indices, 'class'].values)
results = exported_pipeline.predict(tpot_data.loc[testing_indices].drop('class', axis=1))
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
            ("make_union(\n"
            "{},\n"
            "{})").format(_indent(_make_branch(operator[1]), 4),
                          _indent(_make_branch(operator[2]), 4))
        )
    else:
        input_name, args = operator[1], operator[2:]
        tpot_op = operators.Operator.get_by_name(op_name)

        if input_name != 'input_df':
            steps.extend(process_operator(input_name, depth + 1))

        # If the step is a classifier and is not the last step then we must
        # add its guess as a synthetic feature
        if tpot_op.type == "Classifier" and depth > 0:
            steps.append(
                "make_union(VotingClassifier(estimators=[(\"clf\", {})]), FunctionTransformer(lambda X: X))".
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


def _make_branch(branch):
    if branch[1] == "input_df":  # If depth of branch == 1
        tpot_op = operators.Operator.get_by_name(branch[0])

        if tpot_op.type == "Classifier":
            return """make_union(VotingClassifier(estimators=[('branch',
{}
)]), FunctionTransformer(lambda X: X))""".format(_indent(process_operator(branch)[0], 4))
        else:
            return process_operator(branch)[0]
    else:
        return """make_union(VotingClassifier(estimators=[('branch',
{}
)]), FunctionTransformer(lambda X: X))""".format(_indent(generate_pipeline_code(branch), 4))

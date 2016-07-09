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
from .operators import *


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
    # Unroll the nested function calls into serial code. Check export_utils.py for details.
    pipeline_list = unroll_nested_fuction_calls(exported_pipeline)

    # Have the exported code import all of the necessary modules and functions
    pipeline_text = generate_import_code(pipeline_list)

    # Replace the function calls with their corresponding Python code. Check export_utils.py for details.
    pipeline_text += generate_pipeline_code(pipeline_list)

    return pipeline_text


def unroll_nested_fuction_calls(exported_pipeline):
    """Unroll the nested function calls into serial code for use in TPOT.export()

    Parameters
    ----------
    exported_pipeline: deap.creator.Individual
       The pipeline that is being exported

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

    # Always start with these imports
    pipeline_imports = {
        'sklearn.cross_validation': ['train_test_split'],
        'sklearn.pipeline':         ['Pipeline']
    }

    # Build dict of import requirments from list of operators
    import_relations = {}
    for op in Operator.inheritors():
        import_relations[op.__name__] = op.import_hash

    # Build import dict from operators used
    for op in operators_used:
        def merge_imports(old_dict, new_dict):
            # Key is a module name
            for key in new_dict.keys():
                if key in old_dict.keys():
                    # Union imports from the same module
                    old_dict[key] = old_dict[key] | set(new_dict[key])
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


def generate_pipeline_code(pipeline_list):
    """Generate code specific to the construction and execution of the sklearn
    Pipeline

    Parameters
    ----------
    pipeline_list: list
        List of operators in the current optimized pipeline

    Returns
    -------
    pipeline_text: str
        Source code containing code related to the sklearn Pipeline

    """
    steps = []
    for i, operator in enumerate(pipeline_list):
        tpot_op = Operator.get_by_name(operator[1])
        step_name = tpot_op.__class__.__bases__[0].__name__

        args = [eval(x) for x in operator[3:]]  # TODO: Don't use eval()
        steps.append("(\"{}-{}\", {})".format(i, step_name, tpot_op.export(*args)))

    pipeline_text = """
exported_pipeline = Pipeline([
    {STEPS}
])

exported_pipeline.fit(tpot_data.loc[training_indices].drop('class', axis=1).values,
                      tpot_data.loc[training_indices, 'class'].values)
results = exported_pipeline.predict(tpot_data.loc[testing_indices].drop('class', axis=1))
""".format(STEPS=",\n    ".join(steps))

    return pipeline_text

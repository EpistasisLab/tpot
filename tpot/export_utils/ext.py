from . import import_relations, replace_mathematical_operators, unroll_nested_fuction_calls

from jinja2 import Extension, Environment, FileSystemLoader

class PipelineExtension(Extension):
    def __init__(self, environment):
        super(PipelineExtension, self).__init__(environment)
        self.environment = environment
        self.environment.filters['max_feature'] = lambda x: ['auto',None][x] if int(x[3]) <= 1 else 'min({MAX_FEATURES}, len({INPUT_DF}.columns) - 1)'.format(MAX_FEATURES=x[3], INPUT_DF=x[2])

    def preprocess(self, pipeline):
        """
        # Replace all of the mathematical operators with their results. Check export_utils.py for details.
        exported_pipeline = replace_mathematical_operators(exported_pipeline)

        # Unroll the nested function calls into serial code. Check export_utils.py for details.
        exported_pipeline, pipeline_list = unroll_nested_fuction_calls(exported_pipeline)


        # Have the exported code import all of the necessary modules and functions
        pipeline_text = generate_import_code(pipeline_list)

        # Replace the function calls with their corresponding Python code. Check export_utils.py for details.
        pipeline_text += replace_function_calls(pipeline_list)

        """
        pipeline = replace_mathematical_operators(pipeline)
        exported_pipeline, pipeline_list = unroll_nested_fuction_calls(exported_pipeline)
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
                pass # Operator does not require imports
        return self.environment.get_template('source.py').render(pipeline_imports=pipeline_imports)

loader=FileSystemLoader(
    'templates/source.py',
    'templates/*.tmpl.py',
))

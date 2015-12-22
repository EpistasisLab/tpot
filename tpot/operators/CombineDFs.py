from base import BasicOperator
import pandas as pd

class CombineDFs(BasicOperator):
    def __init__(self):
        super(CombineDFs, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, pd.DataFrame], 
            outtype = pd.DataFrame, 
            import_code = '', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._combine_dfs(input_df, *args)
    def callable_code(self, operator_num, operator, result_name):
        operator_text = '''
# Combine two DataFrames
{2} = {0}.join({1}[[column for column in {1}.columns.values if column not in {0}.columns.values]])
'''.format(operator[2], operator[3], result_name)
    
        return operator_text

    @staticmethod
    def _combine_dfs(input_df1, input_df2):
        """Function to combine two DataFrames
        
        Parameters
        ----------
        input_df1: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to combine
        input_df2: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to combine

        Returns
        -------
        combined_df: pandas.DataFrame {n_samples, n_both_features+['guess', 'group', 'class']}
            Returns a DataFrame containing the features of both input_df1 and input_df2

        """
        return input_df1.join(input_df2[[column for column in input_df2.columns.values if column not in input_df1.columns.values]]).copy()
from base import BasicOperator
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class StandardScalerOperator(BasicOperator):
    def __init__(self):
        super(StandardScalerOperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.preprocessing import StandardScaler', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._standard_scaler(input_df)
    def callable_code(self, operator_num, operator, result_name):
        operator_text = '''
# Use Scikit-learn's StandardScaler to scale the features
training_features = {0}.loc[training_indices].drop('class', axis=1)
{1} = {0}.copy()

if len(training_features.columns.values) > 0:
    scaler = StandardScaler()
    scaler.fit(training_features.values.astype(np.float64))
    scaled_features = scaler.transform({1}.drop('class', axis=1).values.astype(np.float64))

    for col_num, column in enumerate({1}.drop('class', axis=1).columns.values):
        {1}.loc[:, column] = scaled_features[:, col_num]
'''.format(operator[2], result_name)
    
        return operator_text

    def _standard_scaler(self, input_df):
        """Uses Scikit-learn's StandardScaler to scale the features by removing their mean and scaling to unit variance

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        scaled_df: pandas.DataFrame {n_samples, n_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the scaled features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # The scaler must be fit on only the training data
        scaler = StandardScaler()
        scaler.fit(training_features.values.astype(np.float64))
        scaled_features = scaler.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        for col_num, column in enumerate(input_df.drop(['class', 'group', 'guess'], axis=1).columns.values):
            input_df.loc[:, column] = scaled_features[:, col_num]

        return input_df.copy()
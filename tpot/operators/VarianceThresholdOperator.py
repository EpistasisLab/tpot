from base import BasicOperator
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

class VarianceThresholdOperator(BasicOperator):
    def __init__(self):
        super(VarianceThresholdOperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, float], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.feature_selection import VarianceThreshold', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._variance_threshold(input_df, *args)
    def callable_code(self, operator_num, operator, result_name):
        operator_text = '''
# Use Scikit-learn's VarianceThreshold for feature selection
training_features = {0}.loc[training_indices].drop('class', axis=1)

selector = VarianceThreshold(threshold={1})
try:
    selector.fit(training_features.values)
except ValueError:
    # None of the features meet the variance threshold
    {2} = {0}[['class']]

mask = selector.get_support(True)
mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
{2} = {0}[mask_cols]
'''.format(operator[2], operator[3], result_name)
    
        return operator_text

    def _variance_threshold(self, input_df, threshold):
        """Uses Scikit-learn's VarianceThreshold feature selection to learn the subset of features that pass the threshold
        
        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        threshold: float
            The variance threshold that removes features that fall under the threshold

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the features that are above the variance threshold

        """

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values

        selector = VarianceThreshold(threshold=threshold)
        try:
            selector.fit(training_features) 
        except ValueError:
            # None features are above the variance threshold
            return input_df[['guess', 'class', 'group']].copy()

        mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()
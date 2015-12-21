from base import BasicOperator
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class SelectKBestOperator(BasicOperator):
    def __init__(self):
        super(SelectKBestOperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.feature_selection import SelectKBest\nfrom sklearn.feature_selection import f_classif', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._select_kbest(input_df, *args)
    def callable_code(self, operator_num, operator, result_name):
        k = int(operator[3])
                
        if k < 1:
            k = 1
        
        k = 'min({}, len(training_features.columns))'.format(k)
        
        operator_text = '''
# Use Scikit-learn's SelectKBest for feature selection
training_features = {0}.loc[training_indices].drop('class', axis=1)
training_class_vals = {0}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {2} = {0}.copy()
else:
    selector = SelectKBest(f_classif, k={1})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {2} = {0}[mask_cols]
'''.format(operator[2], k, result_name)
    
        return operator_text

    def _select_kbest(self, input_df, k):
        """Uses Scikit-learn's SelectKBest feature selection to learn the subset of features that have the highest score according to some scoring function
        
        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        k: int
            The top k features to keep from the original set of features in the training data

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the `k` best features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values
        
        if k < 1:
            k = 1
        elif k >= len(training_features.columns):
            k = 'all'

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        selector = SelectKBest(f_classif, k=k)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()
from base import BasicOperator
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

class RecursiveFeatureElim(BasicOperator):
    def __init__(self):
        super(RecursiveFeatureElim, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, int, float], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.feature_selection import RFE\nfrom sklearn.svm import SVC', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._rfe(input_df, *args)
    def callable_code(self, operator_num, operator, result_name):
        n_features_to_select = int(operator[3])
        step = float(operator[4])
        
        if n_features_to_select < 1:
            n_features_to_select = 1
        n_features_to_select = 'min({}, len(training_features.columns))'.format(n_features_to_select)
        
        if step < 0.1:
            step = 0.1
        elif step >= 1.:
            step = 0.99
        
        operator_text = '''
# Use Scikit-learn's Recursive Feature Elimination (RFE) for feature selection
training_features = {0}.loc[training_indices].drop('class', axis=1)
training_class_vals = {0}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {3} = {0}.copy()
else:
    selector = RFE(SVC(kernel='linear'), n_features_to_select={1}, step={2})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {3} = {0}[mask_cols]
'''.format(operator[2], n_features_to_select, step, result_name)
    
        return operator_text

    def _rfe(self, input_df, num_features, step):
        """Uses Scikit-learn's Recursive Feature Elimination to learn the subset of features that have the highest weights according to the estimator
        
        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        num_features: int
            The number of features to select
        step: float
            The percentage of features to drop each iteration

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the `num_features` best features

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values
        
        if step < 0.1: 
            step = 0.1
        elif step >= 1.:
            step = 0.99
        if num_features < 1:
            num_features = 1
        elif num_features > len(training_features.columns):
            num_features = len(training_features.columns)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        estimator = SVC(kernel='linear')
        selector = RFE(estimator, n_features_to_select=num_features, step=step)
        try:
            selector.fit(training_features, training_class_vals)
            mask = selector.get_support(True)
            mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
            return input_df[mask_cols].copy()
        except ValueError:
            return input_df[['guess', 'class', 'group']].copy()
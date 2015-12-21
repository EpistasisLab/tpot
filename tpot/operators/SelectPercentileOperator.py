# self.pset.addPrimitive(self._select_percentile, [pd.DataFrame, int], pd.DataFrame)
from base import BasicOperator
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

class SelectPercentileOperator(BasicOperator):
    def __init__(self):
        super(SelectPercentileOperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.feature_selection import SelectPercentile\nfrom sklearn.feature_selection import f_classif', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._select_percentile(input_df, *args)
    def callable_code(self, operator_num, operator, result_name):
        percentile = int(operator[3])
                
        if percentile < 0:
            percentile = 0
        elif percentile > 100:
            percentile = 100
        
        operator_text = '''
# Use Scikit-learn's SelectPercentile for feature selection
training_features = {0}.loc[training_indices].drop('class', axis=1)
training_class_vals = {0}.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    {2} = {0}.copy()
else:
    selector = SelectPercentile(f_classif, percentile={1})
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    {2} = {0}[mask_cols]
'''.format(operator[2], percentile, result_name)
    
        return operator_text

    def _select_percentile(self, input_df, percentile):
        """Uses Scikit-learn's SelectPercentile feature selection to learn the subset of features that belong in the highest `percentile`
        according to a given scoring function
        
        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to perform feature selection on
        percentile: int
            The features that belong in the top percentile to keep from the original set of features in the training data

        Returns
        -------
        subsetted_df: pandas.DataFrame {n_samples, n_filtered_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the best features in the given `percentile`

        """
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)
        training_class_vals = input_df.loc[input_df['group'] == 'training', 'class'].values
        
        if percentile < 0: 
            percentile = 0
        elif percentile > 100:
            percentile = 100

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        selector = SelectPercentile(f_classif, percentile=percentile)
        selector.fit(training_features, training_class_vals)
        mask = selector.get_support(True)
        mask_cols = list(training_features.iloc[:, mask].columns) + ['guess', 'class', 'group']
        return input_df[mask_cols].copy()
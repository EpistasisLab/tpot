from base import BasicOperator
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

class PolynomialFeaturesOperator(BasicOperator):
    def __init__(self):
        super(PolynomialFeaturesOperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.preprocessing import PolynomialFeatures', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._polynomial_features(input_df)
    def callable_code(self, operator_num, operator, result_name):
        operator_text = '''
# Use Scikit-learn's PolynomialFeatures to construct new features from the existing feature set
training_features = {0}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0 and len(training_features.columns.values) <= 700:
    # The feature constructor must be fit on only the training data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(training_features.values.astype(np.float64))
    constructed_features = poly.transform({0}.drop('class', axis=1).values.astype(np.float64))

    {0}_classes = {0}['class'].values
    {1} = pd.DataFrame(data=constructed_features)
    {1}['class'] = {0}_classes
else:
    {1} = {0}.copy()
'''.format(operator[2], result_name)
    
        return operator_text

    def _polynomial_features(self, input_df):
        """Uses Scikit-learn's PolynomialFeatures to construct new degree-2 polynomial features from the existing feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_constructed_features + ['guess', 'group', 'class']}
            Returns a DataFrame containing the constructed features

        """
        
        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()
        elif len(training_features.columns.values) > 700:
            # Too many features to produce - skip this operator
            return input_df.copy()

        # The feature constructor must be fit on only the training data
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(training_features.values.astype(np.float64))
        constructed_features = poly.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=constructed_features)
        modified_df['class'] = input_df['class'].values
        modified_df['group'] = input_df['group'].values
        modified_df['guess'] = input_df['guess'].values
        
        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()
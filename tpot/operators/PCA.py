from base import BasicOperator
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class PCAoperator(BasicOperator):
    def __init__(self):
        super(PCAoperator, self).__init__(
            operation_object = None, 
            intypes = [pd.DataFrame, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.decomposition import PCA', 
            callable_code = ''
            )     
    def evaluate_operator(self, input_df, *args, **kargs):
        return self._pca(input_df, *args, **kargs)
    def callable_code(self, operator_num, operator, result_name):
        n_components = int(operator[3])
        if n_components < 1:
            n_components = 1
        n_components = 'min({}, len(training_features.columns.values))'.format(n_components)

        operator_text = '''
# Use Scikit-learn's PCA to transform the feature set
training_features = {0}.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0:
    # PCA must be fit on only the training data
    pca = PCA(n_components={1})
    pca.fit(training_features.values.astype(np.float64))
    transformed_features = pca.transform({0}.drop('class', axis=1).values.astype(np.float64))

    {0}_classes = {0}['class'].values
    {2} = pd.DataFrame(data=transformed_features)
    {2}['class'] = {0}_classes
else:
    {2} = {0}.copy()
'''.format(operator[2], n_components, result_name)
    
        return operator_text

    def _pca(self, input_df, n_components):
        """Uses Scikit-learn's PCA to transform the feature set

        Parameters
        ----------
        input_df: pandas.DataFrame {n_samples, n_features+['class', 'group', 'guess']}
            Input DataFrame to scale
        n_components: int
            The number of components to keep

        Returns
        -------
        modified_df: pandas.DataFrame {n_samples, n_components + ['guess', 'group', 'class']}
            Returns a DataFrame containing the transformed features

        """

        if n_components < 1:
            n_components = 1
        elif n_components >= len(input_df.columns.values) - 3:
            n_components = None

        training_features = input_df.loc[input_df['group'] == 'training'].drop(['class', 'group', 'guess'], axis=1)

        if len(training_features.columns.values) == 0:
            return input_df.copy()

        # PCA must be fit on only the training data
        pca = PCA(n_components=n_components)
        pca.fit(training_features.values.astype(np.float64))
        transformed_features = pca.transform(input_df.drop(['class', 'group', 'guess'], axis=1).values.astype(np.float64))

        modified_df = pd.DataFrame(data=transformed_features)
        modified_df['class'] = input_df['class'].values
        modified_df['group'] = input_df['group'].values
        modified_df['guess'] = input_df['guess'].values

        new_col_names = {}
        for column in modified_df.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        modified_df.rename(columns=new_col_names, inplace=True)

        return modified_df.copy()
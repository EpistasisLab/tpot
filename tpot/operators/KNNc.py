from base import LearnerOperator
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class KNNc(LearnerOperator):
    def __init__(self):
        super(KNNc, self).__init__(
            operation_object = KNeighborsClassifier, 
            intypes = [pd.DataFrame, int], 
            outtype = pd.DataFrame, 
            import_code   = 'from sklearn.neighbors import KNeighborsClassifier', 
            callable_code = 'KNeighborsClassifier(n_neighbors={})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        n_neighbors = args[0]
        
        training_set_size = len(input_df.loc[input_df['group'] == 'training'])
        
        if n_neighbors < 2:
            n_neighbors = 2
        elif n_neighbors >= training_set_size:
            n_neighbors = training_set_size - 1
         
        args=None
        kargs = {'n_neighbors':n_neighbors}
        return input_df, args, kargs
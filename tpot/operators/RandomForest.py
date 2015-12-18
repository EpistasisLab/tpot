from base import LearnerOperator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RandomForest(LearnerOperator):
    def __init__(self):
        super(RandomForest, self).__init__(
            operation_object = RandomForestClassifier, 
            intypes = [pd.DataFrame, int, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.ensemble import RandomForestClassifier', 
            callable_code = 'RandomForestClassifier(n_estimators={}, max_features={})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        n_estimators, max_features = args
        
        if n_estimators < 1:
            n_estimators = 1
        elif n_estimators > 500:
            n_estimators = 500

        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3
         
        args=None
        kargs = {'n_estimators':n_estimators, 'max_features':max_features, 'random_state':42, n_jobs:-1}
        return input_df, args, kargs
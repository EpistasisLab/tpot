from base import LearnerOperator
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

class GradientBoosting(LearnerOperator):
    def __init__(self):
        super(GradientBoosting, self).__init__(
            operation_object = GradientBoostingClassifier, 
            intypes = [pd.DataFrame, float, int, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.ensemble import GradientBoostingClassifier', 
            callable_code = 'GradientBoostingClassifier(learning_rate={1}, n_estimators={2}, max_depth={3})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        learning_rate, n_estimators, max_depth = args
        
        if learning_rate <= 0.:
            learning_rate = 0.0001

        if n_estimators < 1:
            n_estimators = 1
        elif n_estimators > 500:
            n_estimators = 500

        if max_depth < 1:
            max_depth = None
        
        args=None
        kargs = {'learning_rate':learning_rate, 'n_estimators':n_estimators, 
                 'max_depth':max_depth}
        return input_df, args, kargs
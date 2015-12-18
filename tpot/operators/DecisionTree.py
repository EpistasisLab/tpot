from base import LearnerOperator
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DecisionTree(LearnerOperator):
    def __init__(self):
        super(DecisionTree, self).__init__(
            operation_object = DecisionTreeClassifier, 
            intypes = [pd.DataFrame, int, int], 
            outtype = pd.DataFrame, 
            import_code   = 'from sklearn.tree import DecisionTreeClassifier', 
            callable_code = 'DecisionTreeClassifier(max_features=max_features=min({1}, len({0}.columns) - 1), max_depth={2})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        max_features, max_depth = args
        max_features = int(max_features)
        max_depth    = int(max_depth)

        if max_features < 1:
            max_features = 'auto'
        elif max_features == 1:
            max_features = None
        elif max_features > len(input_df.columns) - 3:
            max_features = len(input_df.columns) - 3

        if max_depth < 1:
            max_depth = None
         
        args=None
        kargs = {'max_features':max_features, 'max_depth':max_depth}
        return input_df, args, kargs
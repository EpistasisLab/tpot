from base import LearnerOperator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RandomForest(LearnerOperator):
    def __init__(self):
        super(RandomForest, self).__init__(
            func = RandomForestClassifier, 
            intypes = [pd.DataFrame, int, int], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.ensemble import RandomForestClassifier', 
            callable_code = 'rfc{} = RandomForestClassifier(n_estimators={}, max_features={})'
            )     
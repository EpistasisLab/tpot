from base import LearnerOperator
from sklearn.linear_model import LogisticRegression
import pandas as pd

class LogisticRegressionGLM(LearnerOperator):
    def __init__(self):
        super(LogisticRegressionGLM, self).__init__(
            operation_object = LogisticRegression, 
            intypes = [pd.DataFrame, float], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.linear_model import LogisticRegression', 
            callable_code = 'LogisticRegression(C={1})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        C = args
        if C <= 0.:
            C = 0.0001
        
        args=None
        kargs = {'C':C}
        return input_df, args, kargs
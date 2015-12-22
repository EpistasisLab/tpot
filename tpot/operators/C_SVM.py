from base import LearnerOperator
from sklearn.svm import SVC
import pandas as pd

class C_SVM(LearnerOperator):
    def __init__(self):
        super(C_SVM, self).__init__(
            operation_object = SVC, 
            intypes = [pd.DataFrame, float], 
            outtype = pd.DataFrame, 
            import_code = 'from sklearn.svm import SVC', 
            callable_code = 'SVC(C={1})'
            )     
    def preprocess_args(self, input_df, *args, **kargs): 
        C = args
        if C <= 0.:
            C = 0.0001
        
        args=None
        kargs = {'C':C}
        return input_df, args, kargs
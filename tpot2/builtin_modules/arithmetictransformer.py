import random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#operations are done along axis
#TODO potentially we could do operations on every combo (mul would be all possible pairs multiplied with each other)
class ArithmeticTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self, function,):
        self.function = function

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        if self.function == "add":
                return np.expand_dims(np.sum(X,1),1)
        elif self.function == "mul_neg_1":
                return X*-1
        elif self.function == "mul":
                return np.expand_dims(np.prod(X,1),1)
        
        elif self.function == "safe_reciprocal":
                results = np.divide(1.0, X.astype(float), out=np.zeros_like(X).astype(float), where=X!=0) #TODO remove astypefloat?
                return results
        
        elif self.function == "eq":
                return np.expand_dims(np.all(X == X[0,:], axis = 1),1).astype(float)

        elif self.function == "ne":
                return 1- np.expand_dims(np.all(X == X[0,:], axis = 1),1).astype(float)

        #TODO these could be "sorted order"
        elif self.function == "ge":
                result = X >= 0
                return  result.astype(float)

        elif self.function == "gt":
                result = X > 0
                return  result.astype(float)
        elif self.function ==  "le":
                result = X <= 0
                return  result.astype(float)
        elif self.function ==  "lt":
                result = X < 0
                return  result.astype(float)

                
        elif self.function ==   "min":
                return np.expand_dims(np.amin(X,1),1)
        elif self.function ==  "max":
                return np.expand_dims(np.amax(X,1),1)

        elif self.function ==  "0":
                return np.zeros((X.shape[0],1))
        elif self.function ==  "1":
                return np.ones((X.shape[0],1))

def issorted(x, rev=False):
    if rev:
        s = sorted(x)
        s.reverse()
        if s == x:
            return True
    else:
        if sorted(x) == x:
            return True

    return False




class AddTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.expand_dims(np.sum(X,1),1)

class mul_neg_1_Transformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return X*-1
    
class MulTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.expand_dims(np.prod(X,1),1)

class SafeReciprocalTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.divide(1.0, X.astype(float), out=np.zeros_like(X).astype(float), where=X!=0) #TODO remove astypefloat?

class EQTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.expand_dims(np.all(X == X[0,:], axis = 1),1).astype(float)

class NETransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return 1- np.expand_dims(np.all(X == X[0,:], axis = 1),1).astype(float)



class GETransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        result = X >= 0
        return  result.astype(float)


class GTTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        result = X > 0
        return  result.astype(float)


class GTTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        result = X > 0
        return  result.astype(float)


class LETransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        result = X <= 0
        return  result.astype(float)


class LTTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        result = X < 0
        return  result.astype(float)


class MinTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.expand_dims(np.amin(X,1),1)



class MaxTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.expand_dims(np.amax(X,1),1)


class ZeroTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.zeros((X.shape[0],1))


class OneTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self):
          pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.ones((X.shape[0],1))


class NTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self, n):
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = np.array(self.transform_helper(np.array(X)))
        if transformed_X.dtype != float:
            transformed_X = transformed_X.astype(float)
        
        return transformed_X

    def transform_helper(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.expand_dims(X,0)
        return np.ones((X.shape[0],1))*self.n

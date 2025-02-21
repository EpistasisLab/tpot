"""
This file is part of the TPOT library.

The current version of TPOT was developed at Cedars-Sinai by:
    - Pedro Henrique Ribeiro (https://github.com/perib, https://www.linkedin.com/in/pedro-ribeiro/)
    - Anil Saini (anil.saini@cshs.org)
    - Jose Hernandez (jgh9094@gmail.com)
    - Jay Moran (jay.moran@cshs.org)
    - Nicholas Matsumoto (nicholas.matsumoto@cshs.org)
    - Hyunjun Choi (hyunjun.choi@cshs.org)
    - Gabriel Ketron (gabriel.ketron@cshs.org)
    - Miguel E. Hernandez (miguel.e.hernandez@cshs.org)
    - Jason Moore (moorejh28@gmail.com)

The original version of TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - Jason Moore (moorejh28@gmail.com)
    - and many more generous open-source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


#operations are done along axis
#TODO potentially we could do operations on every combo (mul would be all possible pairs multiplied with each other)
class ArithmeticTransformer(BaseEstimator,TransformerMixin):

    #functions = ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]
    def __init__(self, function,):
        """
        A transformer that applies a function to the input array along axis 1.
        Parameters
        ----------

        function : str
            The function to apply to the input array. The following functions are supported:
            - 'add' : Add all elements along axis 1
            - 'mul_neg_1' : Multiply all elements along axis 1 by -1
            - 'mul' : Multiply all elements along axis 1
            - 'safe_reciprocal' : Take the reciprocal of all elements along axis 1, with a safe division by zero
            - 'eq' : Check if all elements along axis 1 are equal
            - 'ne' : Check if all elements along axis 1 are not equal
            - 'ge' : Check if all elements along axis 1 are greater than or equal to 0
            - 'gt' : Check if all elements along axis 1 are greater than 0
            - 'le' : Check if all elements along axis 1 are less than or equal to 0
            - 'lt' : Check if all elements along axis 1 are less than 0
            - 'min' : Take the minimum of all elements along axis 1
            - 'max' : Take the maximum of all elements along axis 1
            - '0' : Return an array of zeros
            - '1' : Return an array of ones
        """
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
    def __init__(self):
          """
          A transformer that adds all elements along axis 1.
          """
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
    def __init__(self):
        """
        A transformer that multiplies all elements by -1.
        """
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

    def __init__(self):
        """
        A transformer that multiplies all elements along axis 1.
        """
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

    def __init__(self):
        """
        A transformer that takes the reciprocal of all elements, with a safe division by zero.
        """
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

    def __init__(self):
        """
        A transformer that takes checks if all elements in a row are equal.
        """
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

    def __init__(self):
        """
        A transformer that takes checks if all elements in a row are not equal.
        """  
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

    def __init__(self):
        """
        A transformer that takes checks if all elements in a row are greater than or equal to 0.
        """
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
    def __init__(self):
          """
          A transformer that takes checks if all elements in a row are greater than 0.
          """
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
    def __init__(self):
        """
        A transformer that takes checks if all elements in a row are less than or equal to 0.
        """
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
    def __init__(self):
        """
        A transformer that takes checks if all elements in a row are less than 0.
        """
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
    def __init__(self):
        """
        A transformer that takes the minimum of all elements in a row.
        """
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

    def __init__(self):
          """
          A transformer that takes the maximum of all elements in a row.
          """
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

    def __init__(self):
          """
        A transformer that returns an array of zeros.
          """
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
    def __init__(self):
          """
          A transformer that returns an array of ones.
          """
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

    def __init__(self, n):
        """
        A transformer that returns an array of n.
        """
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

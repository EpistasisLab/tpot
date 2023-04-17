"""This file contains the class definition for all the genetic encoders.
All the genetic encoder classes inherit the Scikit learn BaseEstimator and TransformerMixin classes to follow the Scikit learn paradigm. """

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array

    
class DominantEncoder(BaseEstimator, TransformerMixin):
    """This class contains the function definition for encoding the input features as a Dominant genetic model.
    The encoding used is AA(0)->1, Aa(1)->1, aa(2)->0. """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        Dummy function to fit in with the sklearn API and hence work in pipelines.
        
        Parameters
        ----------
        X : array-like
        """
        return self

    def transform(self, X, y=None):
        """Transform the data by applying the Dominant encoding.
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples (number of individuals)
            and n_components is the number of components (number of features).
        y : None
            Unused
            
        Returns
        -------
        X_transformed: numpy ndarray, {n_samples, n_components}
            The encoded feature set
        """
        X = check_array(X)
        map = {0: 1, 1: 1, 2: 0}
        mapping_function = np.vectorize(lambda i: map[i] if i in map else i)

        X_transformed = mapping_function(X)

        return X_transformed

class RecessiveEncoder(BaseEstimator, TransformerMixin):
    """This class contains the function definition for encoding the input features as a Recessive genetic model.
    The encoding used is AA(0)->0, Aa(1)->1, aa(2)->1. """
    
    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        Dummy function to fit in with the sklearn API and hence work in pipelines.
        
        Parameters
        ----------
        X : array-like
        """
        return self

    def transform(self, X, y=None):
        """Transform the data by applying the Recessive encoding.
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples (number of individuals)
            and n_components is the number of components (number of features).
        y : None
            Unused
            
        Returns
        -------
        X_transformed: numpy ndarray, {n_samples, n_components}
            The encoded feature set
        """
        X = check_array(X)
        map = {0: 0, 1: 1, 2: 1}
        mapping_function = np.vectorize(lambda i: map[i] if i in map else i)

        X_transformed = mapping_function(X)

        return X_transformed

class HeterosisEncoder(BaseEstimator, TransformerMixin):
    """This class contains the function definition for encoding the input features as a Heterozygote Advantage genetic model.
    The encoding used is AA(0)->0, Aa(1)->1, aa(2)->0. """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        Dummy function to fit in with the sklearn API and hence work in pipelines.
        
        Parameters
        ----------
        X : array-like
        """
        return self

    def transform(self, X, y=None):
        """Transform the data by applying the Heterosis encoding.
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples (number of individuals)
            and n_components is the number of components (number of features).
        y : None
            Unused
            
        Returns
        -------
        X_transformed: numpy ndarray, {n_samples, n_components}
            The encoded feature set
        """
        X = check_array(X)
        map = {0: 0, 1: 1, 2: 0}
        mapping_function = np.vectorize(lambda i: map[i] if i in map else i)

        X_transformed = mapping_function(X)

        return X_transformed

class UnderDominanceEncoder(BaseEstimator, TransformerMixin):
    """This class contains the function definition for encoding the input features as a Under Dominance genetic model.
    The encoding used is AA(0)->2, Aa(1)->0, aa(2)->1. """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        Dummy function to fit in with the sklearn API and hence work in pipelines.
        
        Parameters
        ----------
        X : array-like
        """
        return self

    def transform(self, X, y=None):
        """Transform the data by applying the Heterosis encoding.
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples (number of individuals)
            and n_components is the number of components (number of features).
        y : None
            Unused
            
        Returns
        -------
        X_transformed: numpy ndarray, {n_samples, n_components}
            The encoded feature set
        """
        X = check_array(X)
        map = {0: 2, 1: 0, 2: 1}
        mapping_function = np.vectorize(lambda i: map[i] if i in map else i)

        X_transformed = mapping_function(X)

        return X_transformed


class OverDominanceEncoder(BaseEstimator, TransformerMixin):
    """This class contains the function definition for encoding the input features as a Over Dominance genetic model.
    The encoding used is AA(0)->1, Aa(1)->2, aa(2)->0. """

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        Dummy function to fit in with the sklearn API and hence work in pipelines.
        
        Parameters
        ----------
        X : array-like
        """
        return self

    def transform(self, X, y=None):
        """Transform the data by applying the Heterosis encoding.
        
        Parameters
        ----------
        X : numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples (number of individuals)
            and n_components is the number of components (number of features).
        y : None
            Unused
            
        Returns
        -------
        X_transformed: numpy ndarray, {n_samples, n_components}
            The encoded feature set
        """
        X = check_array(X)
        map = {0: 1, 1: 2, 2: 0}
        mapping_function = np.vectorize(lambda i: map[i] if i in map else i)

        X_transformed = mapping_function(X)

        return X_transformed






import tpot2
import numpy as np
import pandas as pd
import sklearn
from tpot2 import config
from typing import Generator, List, Tuple, Union
import random
from sklearn.base import BaseEstimator


class SklearnIndividual(tpot2.BaseIndividual):

    def __init__(self,) -> None:
        super().__init__()

    def mutate(self, rng=None):
        return
    
    def crossover(self, other, rng=None):
        return

    def export_pipeline(self) -> BaseEstimator:
        return
    
    def unique_id(self):
        return self
    

class SklearnIndividualGenerator():
    def __init__(self,):
        pass

    def generate(self, rng=None) -> SklearnIndividual:
        pass
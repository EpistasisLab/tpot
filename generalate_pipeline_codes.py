from tpot import TPOTClassifier, TPOTRegressor
from tpot.base import TPOTBase
from tpot.driver import positive_integer, float_range
from tpot.export_utils import export_pipeline, generate_import_code, _indent, generate_pipeline_code
from tpot.decorators import _gp_new_generation
from tpot.gp_types import Output_DF

from tpot.operators import Operator
from tpot.operators.selectors import TPOTSelectKBest

import numpy as np
import inspect
import random
from datetime import datetime

from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split

from deap import creator
from tqdm import tqdm



pipe_tree = ['LogisticRegression', ['PolynomialFeatures', ['CombineDFs', 'input_matrix', 'input_matrix']], 0.72999999999999998, 29, True]

pipe_code = generate_pipeline_code(pipe_tree)
print(pipe_code)


pipe_tree = ['ExtraTreesClassifier', ['PolynomialFeatures', 'input_matrix'], 11, 46.0]



pipe_code = generate_pipeline_code(pipe_tree)
print(pipe_code)


pipe_tree = ['LogisticRegression', ['PolynomialFeatures', 'input_matrix'], 0.72999999999999998, 29, True]



pipe_code = generate_pipeline_code(pipe_tree)
print(pipe_code)


from tpot2.builtin_modules import ArithmeticTransformer, FeatureSetSelector
from functools import partial
import pandas as pd
import numpy as np
from tpot2.builtin_modules import AddTransformer, mul_neg_1_Transformer, MulTransformer, SafeReciprocalTransformer, EQTransformer, NETransformer, GETransformer, GTTransformer, LETransformer, LTTransformer, MinTransformer, MaxTransformer, ZeroTransformer, OneTransformer, NTransformer

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

def get_ArithmeticTransformer_ConfigurationSpace():
        return ConfigurationSpace(
                space = {
                        'function': Categorical("function", ["add", "mul_neg_1", "mul", "safe_reciprocal", "eq","ne","ge","gt","le","lt", "min","max","0","1"]),
                }
        )




# AddTransformer: {}
# mul_neg_1_Transformer: {}
# MulTransformer: {}
# SafeReciprocalTransformer: {}
# EQTransformer: {}
# NETransformer: {}
# GETransformer: {}
# GTTransformer: {}
# LETransformer: {}
# LTTransformer: {}
# MinTransformer: {}
# MaxTransformer: {}


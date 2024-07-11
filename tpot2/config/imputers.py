from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal

simple_imputer_cs = ConfigurationSpace(
    space = {
        'strategy' : Categorical('strategy', ['mean','median', 'most_frequent', ]),
        'add_indicator' : Categorical('add_indicator', [True, False]), 
    }
)
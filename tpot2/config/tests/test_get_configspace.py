import pytest
import tpot2
from sklearn.datasets import load_iris
import random
import sklearn

import tpot2.config

from ..get_configspace import STRING_TO_CLASS

def test_loop_through_all_hyperparameters():

    n_classes=3
    n_samples=100
    n_features=100
    random_state=None

    for class_name, _ in STRING_TO_CLASS.items():
        estnode_gen = tpot2.config.get_search_space(class_name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)

        #generate 10 random hyperparameters and make sure they are all valid
        for i in range(10):
            estnode = estnode_gen.generate()
            est = estnode.export_pipeline()
    
    
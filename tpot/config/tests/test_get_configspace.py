import pytest
import tpot
import sys
from sklearn.datasets import load_iris
import random
import sklearn

import tpot.config

from ..get_configspace import STRING_TO_CLASS, GROUPNAMES

def test_loop_through_all_hyperparameters():

    n_classes=3
    n_samples=100
    n_features=100
    random_state=None

    for class_name, _ in STRING_TO_CLASS.items():
        print(class_name)
        estnode_gen = tpot.config.get_search_space(class_name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)

        #generate 100 random hyperparameters and make sure they are all valid
        for i in range(25):
            estnode = estnode_gen.generate()
            est = estnode.export_pipeline()
    
@pytest.mark.skipif(sys.platform == 'darwin', reason="sklearnex dependency not available on macOS")
def test_loop_through_groupnames():

    n_classes=3
    n_samples=100
    n_features=100
    random_state=None

    for groupname, group in GROUPNAMES.items():
        for class_name in group:
            print(class_name)
            estnode_gen = tpot.config.get_search_space(class_name, n_classes=n_classes, n_samples=n_samples, n_features=n_features, random_state=random_state)

            #generate 10 random hyperparameters and make sure they are all valid
            for i in range(25):
                estnode = estnode_gen.generate()
                est = estnode.export_pipeline()
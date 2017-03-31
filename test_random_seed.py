# coding: utf-8

import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import time
import sys


file_link = 'https://github.com/EpistasisLab/penn-ml-benchmarks/raw/master/datasets/Hill_Valley_without_noise/Hill_Valley_without_noise.csv.gz'
input_data = np.recfromcsv(file_link, delimiter='\t', dtype=np.float64, case_sensitive=True)
features = np.delete(input_data.view(np.float64).reshape(input_data.size, -1),
                     input_data.dtype.names.index('class'), axis=1)

training_features, testing_features, training_classes, testing_classes = train_test_split(features, input_data['class'], random_state=10)

for i in range(10):
    # test a pmlb dataset
    sys.stdout = open('tpot_test_with_xgboost_032917_no_random_seed_in_varOR_{}.txt'.format(i+1), 'w')
    tpot = TPOTClassifier(population_size=20, random_state=10, generations=5, verbosity=2, n_jobs=2)
    tpot.fit(training_features, training_classes)
    print("\n##Test # {} with a fixed random seed of 10 for Hill_Valley_without_noise: {}\n".format(i+1, tpot.score(testing_features,testing_classes)))
    time.sleep(1)

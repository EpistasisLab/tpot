import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))


# Perform classification with a C-support vector classifier
svc2 = SVC(C=13.4)
svc2.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result2 = result1
result2['svc2-classification'] = svc2.predict(result2.drop('class', axis=1).values)

import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))


# Perform classification with a C-support vector classifier
svc3 = SVC(C=2.31034482759)
svc3.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)
result3 = result2
result3['svc3-classification'] = svc3.predict(result3.drop('class', axis=1).values)

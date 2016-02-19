import numpy as np
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = next(iter(StratifiedShuffleSplit(tpot_data['class'].values, n_iter=1, train_size=0.75, test_size=0.25)))


# Perform classification with a random forest classifier
rfc2 = RandomForestClassifier(n_estimators=32, max_features=min(9, len(result1.columns) - 1))
rfc2.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result2 = result1
result2['rfc2-classification'] = rfc2.predict(result2.drop('class', axis=1).values)

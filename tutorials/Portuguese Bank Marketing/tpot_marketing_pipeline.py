import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)

result1 = tpot_data.copy()

# Perform classification with a passive aggressive classifier
pagr1 = PassiveAggressiveClassifier(C=0.81, loss="squared_hinge", fit_intercept=True, random_state=42)
pagr1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)

result1['pagr1-classification'] = pagr1.predict(result1.drop('class', axis=1).values)
